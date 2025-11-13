use std::sync::Once;

use env_logger::Builder;
use pyo3::prelude::*;

use crate::{
    autograd::{self, ForwardModeResult, ReverseModeResult},
    interface::TorchDiffsol,
};

static INIT_LOGGER: Once = Once::new();

#[pyclass(name = "DiffsolModule")]
pub struct PyTorchDiffsol {
    inner: TorchDiffsol,
}

#[pymethods]
impl PyTorchDiffsol {
    #[new]
    fn new(code: &str) -> PyResult<Self> {
        let inner = TorchDiffsol::from_diffsl(code).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Solve the ODE for raw `f64` parameters and times.
    ///
    /// Returns a tuple `(nout, ntimes, flat_solution)` in column-major order.
    #[pyo3(signature = (params, times))]
    fn solve_dense(&self, params: Vec<f64>, times: Vec<f64>) -> PyResult<(usize, usize, Vec<f64>)> {
        let (nout, nt, data) = self
            .inner
            .solve_dense(&params, &times)
            .map_err(PyErr::from)?;
        log::info!(
            "solve_dense completed (params={}, times={}, nout={}, nt={})",
            params.len(),
            times.len(),
            nout,
            nt
        );
        Ok((nout, nt, data))
    }

    /// Run the differentiable forward-mode solver.
    ///
    /// Returns the dense solution followed by stacked sensitivities:
    /// `(nout, nt, sol_flat, nsens, nrows, nt, sens_flat)`, where the
    /// sensitivity buffer is laid out as `nsens` column-major matrices.
    #[pyo3(signature = (params, times))]
    fn forward_mode(
        &self,
        params: Vec<f64>,
        times: Vec<f64>,
    ) -> PyResult<(usize, usize, Vec<f64>, usize, usize, usize, Vec<f64>)> {
        let ForwardModeResult {
            solution,
            sensitivities,
        } = autograd::solve_forward_mode(&self.inner, &params, &times).map_err(PyErr::from)?;
        let (nsens, sens_rows, sens_cols, sens_data) = sensitivities;
        log::info!(
            "forward_mode completed (params={}, times={}, sensitivities={})",
            params.len(),
            times.len(),
            nsens
        );
        Ok((
            solution.nrows,
            solution.ncols,
            solution.data,
            nsens,
            sens_rows,
            sens_cols,
            sens_data,
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (code, params, times, grad_output))]
fn reverse_mode(
    code: &str,
    params: Vec<f64>,
    times: Vec<f64>,
    grad_output: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let module = TorchDiffsol::from_diffsl(code).map_err(PyErr::from)?;
    let ReverseModeResult {
        grad_params,
        grad_initial_state,
    } = autograd::solve_reverse_mode(&module, &params, &times, &grad_output)
        .map_err(PyErr::from)?;
    let mut data = grad_params;
    let mut init = grad_initial_state;
    data.append(&mut init);
    log::info!(
        "reverse_mode completed (params={}, times={}, grad_params={})",
        params.len(),
        times.len(),
        data.len()
    );
    Ok(data)
}

#[pyfunction]
#[pyo3(signature = (level=None))]
fn init_logging(level: Option<&str>) -> PyResult<()> {
    let requested = level.unwrap_or("info").to_string();
    let level_for_init = requested.clone();
    INIT_LOGGER.call_once(move || {
        let mut builder = Builder::new();
        builder.parse_filters(&level_for_init);
        builder.format_timestamp_millis();
        let _ = builder.try_init();
    });
    log::info!("diffsol logging enabled (level={})", requested);
    Ok(())
}

#[pymodule]
fn diffsol_pytorch(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTorchDiffsol>()?;
    m.add_function(wrap_pyfunction!(reverse_mode, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    Ok(())
}
