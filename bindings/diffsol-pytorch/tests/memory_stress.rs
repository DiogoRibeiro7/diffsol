#![cfg(feature = "diagnostics")]

#[cfg(feature = "llvm")]
use diffsol_pytorch::autograd;
use diffsol_pytorch::{diagnostics::leaks::LeakDetector, interface::TorchDiffsol};

const LOGISTIC: &str = r#"
in = [k]
k { 0.5 }
u {
    u = 1.0,
}
F {
    -k * u,
}
"#;

fn times() -> Vec<f64> {
    (0..64).map(|i| i as f64 * 0.05).collect()
}

#[test]
fn dense_solves_do_not_leak_buffers() {
    let params = vec![0.5];
    let times = times();
    let module = TorchDiffsol::from_diffsl(LOGISTIC).expect("failed to build module");
    let guard = LeakDetector::new("dense-solves");
    for _ in 0..64 {
        let (_nout, nt, flat) = module.solve_dense(&params, &times).unwrap();
        assert_eq!(flat.len(), nt * 1);
    }
    guard.assert_clean();
}

#[cfg(feature = "llvm")]
#[test]
fn forward_mode_long_run() {
    let params = vec![0.25];
    let times = times();
    let module = TorchDiffsol::from_diffsl(LOGISTIC).expect("failed to build module");
    let guard = LeakDetector::new("forward-mode");
    for _ in 0..16 {
        let _ = autograd::solve_forward_mode(&module, &params, &times).unwrap();
    }
    guard.assert_clean();
}

#[cfg(feature = "llvm")]
#[test]
fn reverse_mode_does_not_leak_checkpoints() {
    let params = vec![0.4];
    let times = times();
    let module = TorchDiffsol::from_diffsl(LOGISTIC).expect("failed to build module");
    let mut grad_output = vec![0.0; times.len()];
    if let Some(last) = grad_output.last_mut() {
        *last = 1.0;
    }
    let guard = LeakDetector::new("reverse-mode");
    for _ in 0..32 {
        let grads = autograd::solve_reverse_mode(&module, &params, &times, &grad_output).unwrap();
        assert_eq!(grads.grad_params.len(), params.len());
    }
    guard.assert_clean();
}
