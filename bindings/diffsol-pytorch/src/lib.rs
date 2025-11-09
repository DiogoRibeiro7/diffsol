//! PyTorch integration layer for diffsol.

pub mod autograd;
pub mod error;
pub mod interface;

pub use autograd::{solve_forward_mode, solve_reverse_mode, ForwardModeResult, ReverseModeResult};
pub use interface::TorchDiffsol;

#[cfg(feature = "python")]
mod python;
