use std::error::Error;
use std::fmt::{self, Display, Formatter};

use diffsol::error::DiffsolError;
use thiserror::Error;

pub type TorchResult<T> = Result<T, TorchDiffsolError>;

type BoxError = Box<dyn Error + Send + Sync + 'static>;

#[derive(Debug, Clone, Copy)]
pub enum BuildStage {
    ParseModule,
    InitializeSolver,
    PythonBinding,
}

#[derive(Debug, Clone, Copy)]
pub enum AutodiffStage {
    ForwardSetup,
    ForwardSolve,
    ReverseSetup,
    ReverseSolve,
}

#[derive(Debug, Clone, Copy)]
pub enum RuntimeStage {
    SolverSetup,
    SolveDense,
    EvaluateOutput,
    PythonBinding,
}

impl Display for BuildStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BuildStage::ParseModule => write!(f, "DiffSL parsing/compilation"),
            BuildStage::InitializeSolver => write!(f, "solver initialisation"),
            BuildStage::PythonBinding => write!(f, "Python binding initialisation"),
        }
    }
}

impl Display for AutodiffStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            AutodiffStage::ForwardSetup => write!(f, "forward-mode setup"),
            AutodiffStage::ForwardSolve => write!(f, "forward-mode solve"),
            AutodiffStage::ReverseSetup => write!(f, "reverse-mode setup"),
            AutodiffStage::ReverseSolve => write!(f, "reverse-mode solve"),
        }
    }
}

impl Display for RuntimeStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeStage::SolverSetup => write!(f, "solver setup"),
            RuntimeStage::SolveDense => write!(f, "dense solve"),
            RuntimeStage::EvaluateOutput => write!(f, "observable evaluation"),
            RuntimeStage::PythonBinding => write!(f, "python binding call"),
        }
    }
}

#[derive(Debug, Error)]
pub enum TorchDiffsolError {
    #[error("build failure during {stage}: {message}")]
    Build {
        stage: BuildStage,
        message: String,
        #[source]
        source: Option<BoxError>,
        suggestion: Option<&'static str>,
    },
    #[error("autodiff error during {stage}: {message}")]
    Autodiff {
        stage: AutodiffStage,
        message: String,
        #[source]
        source: Option<BoxError>,
        suggestion: Option<&'static str>,
    },
    #[error("runtime error during {stage}: {message}")]
    Runtime {
        stage: RuntimeStage,
        message: String,
        #[source]
        source: Option<BoxError>,
        suggestion: Option<&'static str>,
    },
    #[error("shape mismatch: expected {expected}, got {found}")]
    Shape {
        expected: &'static str,
        found: String,
        suggestion: Option<&'static str>,
    },
}

impl TorchDiffsolError {
    pub fn build(stage: BuildStage, message: impl Into<String>) -> Self {
        TorchDiffsolError::Build {
            stage,
            message: message.into(),
            source: None,
            suggestion: None,
        }
    }

    pub fn autodiff(stage: AutodiffStage, message: impl Into<String>) -> Self {
        TorchDiffsolError::Autodiff {
            stage,
            message: message.into(),
            source: None,
            suggestion: None,
        }
    }

    pub fn runtime(stage: RuntimeStage, message: impl Into<String>) -> Self {
        TorchDiffsolError::Runtime {
            stage,
            message: message.into(),
            source: None,
            suggestion: None,
        }
    }

    pub fn shape(expected: &'static str, found: impl Into<String>) -> Self {
        TorchDiffsolError::Shape {
            expected,
            found: found.into(),
            suggestion: None,
        }
    }

    pub fn build_from_diffsol(stage: BuildStage, err: DiffsolError) -> Self {
        TorchDiffsolError::build(stage, err.to_string()).with_source(err)
    }

    pub fn autodiff_from_diffsol(stage: AutodiffStage, err: DiffsolError) -> Self {
        TorchDiffsolError::autodiff(stage, err.to_string()).with_source(err)
    }

    pub fn runtime_from_diffsol(stage: RuntimeStage, err: DiffsolError) -> Self {
        TorchDiffsolError::runtime(stage, err.to_string()).with_source(err)
    }

    pub fn with_source<E>(mut self, source: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        match &mut self {
            TorchDiffsolError::Build {
                source: ref mut slot,
                ..
            } => {
                *slot = Some(Box::new(source));
            }
            TorchDiffsolError::Autodiff {
                source: ref mut slot,
                ..
            } => {
                *slot = Some(Box::new(source));
            }
            TorchDiffsolError::Runtime {
                source: ref mut slot,
                ..
            } => {
                *slot = Some(Box::new(source));
            }
            TorchDiffsolError::Shape { .. } => {}
        }
        self
    }

    pub fn with_suggestion(mut self, suggestion: &'static str) -> Self {
        match &mut self {
            TorchDiffsolError::Build {
                suggestion: slot, ..
            } => *slot = Some(suggestion),
            TorchDiffsolError::Autodiff {
                suggestion: slot, ..
            } => *slot = Some(suggestion),
            TorchDiffsolError::Runtime {
                suggestion: slot, ..
            } => *slot = Some(suggestion),
            TorchDiffsolError::Shape {
                suggestion: slot, ..
            } => *slot = Some(suggestion),
        }
        self
    }

    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            TorchDiffsolError::Build { suggestion, .. }
            | TorchDiffsolError::Autodiff { suggestion, .. }
            | TorchDiffsolError::Runtime { suggestion, .. }
            | TorchDiffsolError::Shape { suggestion, .. } => *suggestion,
        }
    }
}

impl From<std::io::Error> for TorchDiffsolError {
    fn from(value: std::io::Error) -> Self {
        TorchDiffsolError::runtime(RuntimeStage::PythonBinding, value.to_string())
            .with_source(value)
    }
}

pub trait DiffsolResultExt<T> {
    fn map_build(self, stage: BuildStage, suggestion: Option<&'static str>) -> TorchResult<T>;
    fn map_autodiff(self, stage: AutodiffStage, suggestion: Option<&'static str>)
        -> TorchResult<T>;
    fn map_runtime(self, stage: RuntimeStage, suggestion: Option<&'static str>) -> TorchResult<T>;
}

impl<T> DiffsolResultExt<T> for Result<T, DiffsolError> {
    fn map_build(self, stage: BuildStage, suggestion: Option<&'static str>) -> TorchResult<T> {
        self.map_err(|err| {
            let base = TorchDiffsolError::build_from_diffsol(stage, err);
            match suggestion {
                Some(help) => base.with_suggestion(help),
                None => base,
            }
        })
    }

    fn map_autodiff(
        self,
        stage: AutodiffStage,
        suggestion: Option<&'static str>,
    ) -> TorchResult<T> {
        self.map_err(|err| {
            let base = TorchDiffsolError::autodiff_from_diffsol(stage, err);
            match suggestion {
                Some(help) => base.with_suggestion(help),
                None => base,
            }
        })
    }

    fn map_runtime(self, stage: RuntimeStage, suggestion: Option<&'static str>) -> TorchResult<T> {
        self.map_err(|err| {
            let base = TorchDiffsolError::runtime_from_diffsol(stage, err);
            match suggestion {
                Some(help) => base.with_suggestion(help),
                None => base,
            }
        })
    }
}

#[cfg(feature = "python")]
impl From<TorchDiffsolError> for pyo3::PyErr {
    fn from(err: TorchDiffsolError) -> Self {
        use pyo3::exceptions::{PyRuntimeError, PyValueError};

        let mut message = err.to_string();
        if let Some(help) = err.suggestion() {
            message.push_str(&format!("\n\nHelp: {help}"));
        }
        match err {
            TorchDiffsolError::Shape { .. } => PyValueError::new_err(message),
            TorchDiffsolError::Build { .. }
            | TorchDiffsolError::Autodiff { .. }
            | TorchDiffsolError::Runtime { .. } => PyRuntimeError::new_err(message),
        }
    }
}
