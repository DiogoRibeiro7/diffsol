use diffsol::{
    ode_solver::{
        adjoint::AdjointOdeSolverMethod, sensitivities::SensitivitiesOdeSolverMethod,
        state::StateCommon,
    },
    DenseMatrix, IndexType, MatrixCommon, OdeEquations, OdeSolverMethod, OdeSolverState, Op,
    Vector as VectorTrait, VectorHost,
};

use crate::{
    error::{TorchDiffsolError, TorchResult},
    interface::{Context, LinearSolver, StateMatrix, StateVector, TorchDiffsol},
};

pub struct MatrixData {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<f64>,
}

pub struct ForwardModeResult {
    pub solution: MatrixData,
    pub sensitivities: (usize, usize, usize, Vec<f64>),
}

pub struct ReverseModeResult {
    pub grad_params: Vec<f64>,
    pub grad_initial_state: Vec<f64>,
}

pub fn solve_forward_mode(
    model: &TorchDiffsol,
    params: &[f64],
    times: &[f64],
) -> TorchResult<ForwardModeResult> {
    if times.is_empty() {
        return Err(TorchDiffsolError::Shape(
            "time grid must contain at least one point".to_string(),
        ));
    }

    let (solution, sensitivities) = model.with_problem(|problem| {
        let ctx = problem.eqn.context().clone();
        let params_v = StateVector::from_slice(params, ctx.clone());
        problem.eqn.set_params(&params_v);
        let mut solver = problem.bdf_sens::<LinearSolver>()?;
        let (sol, sens) = solver.solve_dense_sensitivities(times)?;
        Ok((sol, sens))
    })?;

    let sol_data = MatrixData {
        nrows: solution.nrows() as usize,
        ncols: solution.ncols() as usize,
        data: dense_to_vec(&solution),
    };
    let nsens = sensitivities.len();
    let nout = if nsens > 0 {
        sensitivities[0].nrows() as usize
    } else {
        0
    };
    let ntimes = if nsens > 0 {
        sensitivities[0].ncols() as usize
    } else {
        times.len()
    };
    let mut sens_buf = vec![0f64; nsens * nout * ntimes];
    for (i, mat) in sensitivities.iter().enumerate() {
        let data = dense_to_vec(mat);
        let start = i * nout * ntimes;
        sens_buf[start..start + nout * ntimes].copy_from_slice(&data);
    }

    Ok(ForwardModeResult {
        solution: sol_data,
        sensitivities: (nsens, nout, ntimes, sens_buf),
    })
}

pub fn solve_reverse_mode(
    model: &TorchDiffsol,
    params: &[f64],
    times: &[f64],
    grad_output: &[f64],
    nout: usize,
) -> TorchResult<ReverseModeResult> {
    if times.is_empty() {
        return Err(TorchDiffsolError::Shape(
            "time grid must contain at least one point".to_string(),
        ));
    }
    if grad_output.len() % times.len() != 0 {
        return Err(TorchDiffsolError::Shape(
            "gradient output length must match number of time points".to_string(),
        ));
    }
    let grad_matrix = matrix_from_vec(nout, times.len(), grad_output)
        .ok_or_else(|| TorchDiffsolError::Shape("invalid gradient dimensions".to_string()))?;

    let (grad_params, grad_initial): (Vec<StateVector>, StateVector) =
        model.with_problem(|problem| {
            let ctx = problem.eqn.context().clone();
            let params_v = StateVector::from_slice(params, ctx.clone());
            problem.eqn.set_params(&params_v);
            let mut forward = problem.bdf::<LinearSolver>()?;
            let (checkpointing, _sol) = forward.solve_dense_with_checkpointing(times, None)?;
            let adjoint =
                problem.bdf_solver_adjoint::<LinearSolver, _>(checkpointing, Some(nout))?;
            let grad_refs: Vec<&StateMatrix> = vec![&grad_matrix];
            let state = adjoint.solve_adjoint_backwards_pass(times, &grad_refs)?;
            let StateCommon { sg, y, .. } = state.into_common();
            Ok((sg, y))
        })?;

    let grad_params_vec = if grad_params.is_empty() {
        Vec::new()
    } else {
        grad_params[0].as_slice().to_vec()
    };
    let grad_initial_vec = grad_initial.as_slice().to_vec();

    Ok(ReverseModeResult {
        grad_params: grad_params_vec,
        grad_initial_state: grad_initial_vec,
    })
}

fn dense_to_vec(mat: &StateMatrix) -> Vec<f64> {
    let nrows = mat.nrows() as usize;
    let ncols = mat.ncols() as usize;
    let mut buffer = vec![0f64; nrows * ncols];
    for col in 0..ncols {
        for row in 0..nrows {
            buffer[col * nrows + row] = mat[(row as IndexType, col as IndexType)];
        }
    }
    buffer
}

fn matrix_from_vec(nrows: usize, ncols: usize, data: &[f64]) -> Option<StateMatrix> {
    if data.len() != nrows * ncols {
        return None;
    }
    Some(StateMatrix::from_vec(
        nrows,
        ncols,
        data.to_vec(),
        Context::default(),
    ))
}
