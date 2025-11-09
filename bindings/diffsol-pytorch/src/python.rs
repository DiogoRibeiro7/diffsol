use pyo3::prelude::*;

use crate::{
    autograd::{self, ForwardModeResult, ReverseModeResult},
    interface::TorchDiffsol,
};

#[pyclass(name = "DiffsolModule")]
pub struct PyTorchDiffsol {
    inner: TorchDiffsol,
}

#[pymethods]
impl PyTorchDiffsol {
    #[new]
    fn new(code: &str) -> PyResult<Self> {
        let inner = TorchDiffsol::from_diffsl(code)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
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
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
        Ok((nout, nt, data))
    }

    /// Run the differentiable forward-mode solver.
    #[pyo3(signature = (params, times))]
    fn forward_mode(
        &self,
        params: Vec<f64>,
        times: Vec<f64>,
    ) -> PyResult<(usize, usize, Vec<f64>)> {
        let ForwardModeResult { solution, .. } =
            autograd::solve_forward_mode(&self.inner, &params, &times).map_err(|err| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
            })?;
        Ok((solution.nrows, solution.ncols, solution.data))
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
    let module = TorchDiffsol::from_diffsl(code)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    if times.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time grid must be non-empty",
        ));
    }
    let ncols = times.len();
    if grad_output.len() % ncols != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "gradient output length must be a multiple of len(times)",
        ));
    }
    let nrows = grad_output.len() / ncols;
    let ReverseModeResult {
        grad_params,
        grad_initial_state,
    } = autograd::solve_reverse_mode(&module, &params, &times, &grad_output, nrows)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    let mut data = grad_params;
    let mut init = grad_initial_state;
    data.append(&mut init);
    Ok(data)
}

#[pymodule]
fn diffsol_pytorch(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTorchDiffsol>()?;
    m.add_function(wrap_pyfunction!(reverse_mode, m)?)?;
    Ok(())
}
