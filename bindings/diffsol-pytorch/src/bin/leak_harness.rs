#![cfg(feature = "mem-profiling")]

use diffsol_pytorch::{
    autograd,
    diagnostics::{jemalloc_stats, leaks::LeakDetector},
    TorchDiffsol,
};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const LOGISTIC: &str = r#"
in = [k]
k { 0.8 }
u {
    u = 1.0,
}
F {
    -k * u,
}
"#;

fn main() {
    run_harness();
}

fn run_harness() {
    let mut times = Vec::with_capacity(256);
    for i in 0..256 {
        times.push(i as f64 * 0.02);
    }
    let params = vec![0.8];
    let module =
        TorchDiffsol::from_diffsl(LOGISTIC).expect("failed to compile DiffSL problem for harness");

    let before = jemalloc_stats::allocated_bytes();
    {
        let guard = LeakDetector::new("leak-harness");
        exercise_forward(&module, &params, &times);
        #[cfg(feature = "llvm")]
        exercise_reverse(&module, &params, &times);
        guard.assert_clean();
    }
    let after = jemalloc_stats::allocated_bytes();

    const MAX_DRIFT_BYTES: u64 = 32 * 1024;
    if after > before + MAX_DRIFT_BYTES {
        panic!(
            "jemalloc allocated bytes increased by {} (> {} threshold)",
            after.saturating_sub(before),
            MAX_DRIFT_BYTES
        );
    }
}

fn exercise_forward(model: &TorchDiffsol, params: &[f64], times: &[f64]) {
    for _ in 0..128 {
        let (_nout, _nt, buf) = model
            .solve_dense(params, times)
            .expect("forward solve failed");
        assert_eq!(buf.len(), times.len());
    }
}

#[cfg(feature = "llvm")]
fn exercise_reverse(model: &TorchDiffsol, params: &[f64], times: &[f64]) {
    let mut grad_output = vec![0.0; times.len()];
    if let Some(last) = grad_output.last_mut() {
        *last = 1.0;
    }
    for _ in 0..48 {
        let result = autograd::solve_reverse_mode(model, params, times, &grad_output)
            .expect("reverse mode failed");
        assert_eq!(result.grad_params.len(), params.len());
    }
}

#[cfg(not(feature = "llvm"))]
fn exercise_reverse(_model: &TorchDiffsol, _params: &[f64], _times: &[f64]) {}
