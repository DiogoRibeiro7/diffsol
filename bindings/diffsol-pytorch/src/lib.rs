//! PyTorch integration layer for diffsol.

pub mod autograd;
#[cfg(any(test, feature = "diagnostics", feature = "mem-profiling"))]
pub mod diagnostics;
pub mod error;
pub mod interface;
mod memory;

pub use autograd::{solve_forward_mode, solve_reverse_mode, ForwardModeResult, ReverseModeResult};
pub use interface::TorchDiffsol;

#[cfg(feature = "python")]
mod python;
