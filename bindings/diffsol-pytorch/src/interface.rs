use std::sync::Arc;

use parking_lot::Mutex;

use diffsol::{
    NalgebraContext, NalgebraLU, NalgebraMat, NalgebraVec, NonLinearOp, OdeBuilder, OdeEquations,
    OdeSolverMethod, OdeSolverProblem, Op, Vector, VectorHost,
};

#[cfg(not(feature = "llvm"))]
pub(crate) type Codegen = diffsol::CraneliftJitModule;
#[cfg(feature = "llvm")]
pub(crate) type Codegen = diffsol::LlvmModule;

pub(crate) type StateMatrix = NalgebraMat<f64>;
pub(crate) type StateVector = NalgebraVec<f64>;
pub(crate) type Context = NalgebraContext;
pub(crate) type LinearSolver = NalgebraLU<f64>;
pub(crate) type Equation = diffsol::DiffSl<StateMatrix, Codegen>;

use crate::{
    error::{BuildStage, DiffsolResultExt, RuntimeStage, TorchDiffsolError, TorchResult},
    memory::HostBuffer,
};

pub struct TorchDiffsol {
    problem: Arc<Mutex<OdeSolverProblem<Equation>>>,
}

impl Clone for TorchDiffsol {
    fn clone(&self) -> Self {
        Self {
            problem: Arc::clone(&self.problem),
        }
    }
}

impl TorchDiffsol {
    pub fn from_diffsl(code: &str) -> TorchResult<Self> {
        let problem = OdeBuilder::<StateMatrix>::new()
            .build_from_diffsl(code)
            .map_build(
                BuildStage::ParseModule,
                Some("Ensure the DiffSL module compiles with `diffsl` and enable `diffsl-llvm` for autodiff."),
            )?;
        Ok(Self::from_problem(problem))
    }

    pub fn from_problem(problem: OdeSolverProblem<Equation>) -> Self {
        Self {
            problem: Arc::new(Mutex::new(problem)),
        }
    }

    pub fn solve_dense(
        &self,
        params: &[f64],
        times: &[f64],
    ) -> TorchResult<(usize, usize, Vec<f64>)> {
        if times.is_empty() {
            return Err(TorchDiffsolError::shape(
                "at least one requested time",
                "received 0 points",
            )
            .with_suggestion("Provide a non-empty array of evaluation times."));
        }

        let mut problem = self.problem.lock();
        let ctx = problem.eqn.context().clone();
        let params_v = StateVector::from_slice(params, ctx.clone());
        problem.eqn.set_params(&params_v);
        let mut solver = problem.bdf::<LinearSolver>().map_runtime(
            RuntimeStage::SolverSetup,
            Some("Verify the problem definition and initial state are consistent."),
        )?;

        let nout = if let Some(out) = problem.eqn.out() {
            out.nout()
        } else {
            problem.eqn.nstates()
        };

        let mut buffer = HostBuffer::zeros(nout * times.len());
        for (i, &t) in times.iter().enumerate() {
            while solver.state().t < t {
                solver.step().map_runtime(
                    RuntimeStage::SolveDense,
                    Some("Consider loosening tolerances or reviewing event functions."),
                )?;
            }
            let y = solver.interpolate(t).map_runtime(
                RuntimeStage::EvaluateOutput,
                Some("Check that requested times lie within the integration window."),
            )?;
            let out = if let Some(eval) = problem.eqn.out() {
                eval.call(&y, t)
            } else {
                y
            };
            let start = i * nout;
            buffer[start..start + nout].copy_from_slice(out.as_slice());
        }
        drop(problem);

        Ok((nout, times.len(), buffer.into_vec()))
    }

    pub(crate) fn with_problem<F, R>(&self, f: F) -> TorchResult<R>
    where
        F: FnOnce(&mut OdeSolverProblem<Equation>) -> TorchResult<R>,
    {
        let mut guard = self.problem.lock();
        f(&mut guard)
    }
}
