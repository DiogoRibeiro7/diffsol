use diffsol::{
    ode_solver::{
        adjoint::AdjointOdeSolverMethod, sensitivities::SensitivitiesOdeSolverMethod,
        state::StateCommon,
    },
    DenseMatrix, IndexType, MatrixCommon, OdeEquations, OdeSolverMethod, OdeSolverState, Op,
    Vector as VectorTrait, VectorHost,
};

use crate::{
    error::{AutodiffStage, DiffsolResultExt, RuntimeStage, TorchDiffsolError, TorchResult},
    interface::{Context, LinearSolver, StateMatrix, StateVector, TorchDiffsol},
    memory::HostBuffer,
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
        return Err(
            TorchDiffsolError::shape("at least one requested time", "received 0 points")
                .with_suggestion("Provide a non-empty time grid."),
        );
    }

    let (solution, sensitivities) = model.with_problem(|problem| {
        let ctx = problem.eqn.context().clone();
        let params_v = StateVector::from_slice(params, ctx.clone());
        problem.eqn.set_params(&params_v);
        let mut solver = problem.bdf_sens::<LinearSolver>().map_autodiff(
            AutodiffStage::ForwardSetup,
            Some("Ensure the build includes LLVM/Enzyme via `--features diffsl-llvm`."),
        )?;
        let (sol, sens) = solver.solve_dense_sensitivities(times).map_autodiff(
            AutodiffStage::ForwardSolve,
            Some(
                "Check that the problem definition is differentiable for the supplied parameters.",
            ),
        )?;
        Ok((sol, sens))
    })?;

    let sol_data = MatrixData {
        nrows: solution.nrows() as usize,
        ncols: solution.ncols() as usize,
        data: dense_to_vec(&solution).into_vec(),
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
    let sens_vec = if nsens == 0 {
        Vec::new()
    } else {
        let mut sens_buf = HostBuffer::zeros(nsens * nout * ntimes);
        for (i, mat) in sensitivities.iter().enumerate() {
            let data = dense_to_vec(mat);
            let start = i * nout * ntimes;
            sens_buf[start..start + nout * ntimes].copy_from_slice(&data);
        }
        sens_buf.into_vec()
    };

    Ok(ForwardModeResult {
        solution: sol_data,
        sensitivities: (nsens, nout, ntimes, sens_vec),
    })
}

pub fn solve_reverse_mode(
    model: &TorchDiffsol,
    params: &[f64],
    times: &[f64],
    grad_output: &[f64],
) -> TorchResult<ReverseModeResult> {
    if times.is_empty() {
        return Err(
            TorchDiffsolError::shape("at least one requested time", "received 0 points")
                .with_suggestion("Provide a non-empty time grid."),
        );
    }
    if grad_output.is_empty() {
        return Err(TorchDiffsolError::shape(
            "gradient output with solver dimension",
            "received 0 values",
        )
        .with_suggestion("Pass the flattened gradient buffer returned by your loss function."));
    }
    if grad_output.len() % times.len() != 0 {
        return Err(TorchDiffsolError::shape(
            "gradient output with len(times) * nout elements",
            format!("{} values", grad_output.len()),
        )
        .with_suggestion("Flatten the gradient array with shape (nout, len(times))."));
    }
    let nout = grad_output.len() / times.len();
    let grad_matrix = matrix_from_vec(nout, times.len(), grad_output)?;

    let (grad_params, grad_initial): (Vec<StateVector>, StateVector) =
        model.with_problem(|problem| {
            let ctx = problem.eqn.context().clone();
            let params_v = StateVector::from_slice(params, ctx.clone());
            problem.eqn.set_params(&params_v);
            let mut forward = problem
                .bdf::<LinearSolver>()
                .map_runtime(
                    RuntimeStage::SolverSetup,
                    Some("Check that the forward problem can be solved before requesting gradients."),
                )?;
            let (checkpointing, _sol) = forward
                .solve_dense_with_checkpointing(times, None)
                .map_autodiff(
                    AutodiffStage::ReverseSetup,
                    Some("Try reducing the number of checkpointed times or loosening tolerances."),
                )?;
            let adjoint =
                problem
                    .bdf_solver_adjoint::<LinearSolver, _>(checkpointing, Some(nout))
                    .map_autodiff(
                        AutodiffStage::ReverseSetup,
                        Some("Ensure the build enables reverse-mode autodiff and the gradients are well-defined."),
                    )?;
            let grad_refs: Vec<&StateMatrix> = vec![&grad_matrix];
            let state = adjoint
                .solve_adjoint_backwards_pass(times, &grad_refs)
                .map_autodiff(
                    AutodiffStage::ReverseSolve,
                    Some("Check that the gradient output matches the solver's output dimension."),
                )?;
            let StateCommon { sg, y, .. } = state.into_common();
            Ok((sg, y))
        })?;

    let grad_params_vec = if grad_params.is_empty() {
        Vec::new()
    } else {
        let mut buf = HostBuffer::zeros(grad_params[0].len());
        buf.copy_from_slice(grad_params[0].as_slice());
        buf.into_vec()
    };
    let mut grad_initial_buf = HostBuffer::zeros(grad_initial.len());
    grad_initial_buf.copy_from_slice(grad_initial.as_slice());
    let grad_initial_vec = grad_initial_buf.into_vec();

    Ok(ReverseModeResult {
        grad_params: grad_params_vec,
        grad_initial_state: grad_initial_vec,
    })
}

fn dense_to_vec(mat: &StateMatrix) -> HostBuffer {
    let nrows = mat.nrows() as usize;
    let ncols = mat.ncols() as usize;
    let mut buffer = HostBuffer::zeros(nrows * ncols);
    for col in 0..ncols {
        for row in 0..nrows {
            buffer[col * nrows + row] = mat[(row as IndexType, col as IndexType)];
        }
    }
    buffer
}

fn matrix_from_vec(nrows: usize, ncols: usize, data: &[f64]) -> TorchResult<StateMatrix> {
    if data.len() != nrows * ncols {
        return Err(TorchDiffsolError::shape(
            "gradient output with len(times) * nout elements",
            format!("{} values", data.len()),
        )
        .with_suggestion("Flatten the gradient buffer with shape (nout, len(times))."));
    }
    Ok(StateMatrix::from_vec(
        nrows,
        ncols,
        data.to_vec(),
        Context::default(),
    ))
}
