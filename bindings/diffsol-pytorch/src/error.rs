use diffsol::error::DiffsolError;
use thiserror::Error;

pub type TorchResult<T> = Result<T, TorchDiffsolError>;

#[derive(Debug, Error)]
pub enum TorchDiffsolError {
    #[error("diffsol error: {0}")]
    Diffsol(#[from] DiffsolError),
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for TorchDiffsolError {
    fn from(value: std::io::Error) -> Self {
        Self::Other(value.to_string())
    }
}
