#[cfg(any(test, feature = "diagnostics", feature = "mem-profiling"))]
pub mod leaks {
    pub use crate::memory::{snapshot, LeakDetector, MemoryStats};
}

#[cfg(feature = "mem-profiling")]
pub mod jemalloc_stats {
    use tikv_jemalloc_ctl::{epoch, stats};

    pub fn allocated_bytes() -> u64 {
        // Refresh the epoch to fetch up-to-date counters. Errors are non-fatal.
        let _ = epoch::advance();
        stats::allocated::read().unwrap_or(0)
    }
}
