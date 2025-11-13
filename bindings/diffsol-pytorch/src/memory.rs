use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE_BUFFERS: AtomicUsize = AtomicUsize::new(0);
static LIVE_ELEMENTS: AtomicUsize = AtomicUsize::new(0);

/// Heap-backed buffer that tracks outstanding host allocations.
pub(crate) struct HostBuffer {
    data: Vec<f64>,
    accounted: bool,
}

impl HostBuffer {
    pub fn zeros(len: usize) -> Self {
        LIVE_BUFFERS.fetch_add(1, Ordering::SeqCst);
        LIVE_ELEMENTS.fetch_add(len, Ordering::SeqCst);
        Self {
            data: vec![0.0; len],
            accounted: true,
        }
    }

    pub fn into_vec(mut self) -> Vec<f64> {
        self.disarm();
        std::mem::take(&mut self.data)
    }

    fn disarm(&mut self) {
        if self.accounted {
            LIVE_BUFFERS.fetch_sub(1, Ordering::SeqCst);
            LIVE_ELEMENTS.fetch_sub(self.data.len(), Ordering::SeqCst);
            self.accounted = false;
        }
    }
}

impl Deref for HostBuffer {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for HostBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        self.disarm();
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStats {
    pub buffers: usize,
    pub elements: usize,
}

impl MemoryStats {
    pub fn delta(self, other: MemoryStats) -> MemoryStats {
        MemoryStats {
            buffers: self.buffers.saturating_sub(other.buffers),
            elements: self.elements.saturating_sub(other.elements),
        }
    }
}

pub fn snapshot() -> MemoryStats {
    MemoryStats {
        buffers: LIVE_BUFFERS.load(Ordering::SeqCst),
        elements: LIVE_ELEMENTS.load(Ordering::SeqCst),
    }
}

pub struct LeakDetector {
    label: &'static str,
    start: MemoryStats,
    consumed: bool,
}

impl LeakDetector {
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            start: snapshot(),
            consumed: false,
        }
    }

    pub fn assert_clean(mut self) {
        self.consumed = true;
        let end = snapshot();
        let delta = end.delta(self.start);
        assert_eq!(
            delta.buffers, 0,
            "memory leak detected in {} (buffers: +{}, elements: +{})",
            self.label, delta.buffers, delta.elements
        );
        assert_eq!(
            delta.elements, 0,
            "memory leak detected in {} (elements: +{})",
            self.label, delta.elements
        );
    }
}

impl Drop for LeakDetector {
    fn drop(&mut self) {
        if !self.consumed {
            let end = snapshot();
            let delta = end.delta(self.start);
            if delta.buffers != 0 || delta.elements != 0 {
                panic!(
                    "memory leak detected in {} (buffers: +{}, elements: +{})",
                    self.label, delta.buffers, delta.elements
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_buffer_updates_counters() {
        let start = snapshot();
        {
            let guard = LeakDetector::new("host-buffer");
            let mut buf = HostBuffer::zeros(4);
            assert_eq!(buf.len(), 4);
            buf[0] = 1.0;
            let vec = buf.into_vec();
            assert_eq!(vec[0], 1.0);
            guard.assert_clean();
        }
        assert_eq!(snapshot(), start);
    }
}
