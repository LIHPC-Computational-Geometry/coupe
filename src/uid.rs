use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn uid() -> usize {
    ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uid_are_unique() {
        use rayon::iter::ParallelIterator;

        const NUM_UIDS: usize = 10000;

        let mut uids: Vec<usize> = rayon::iter::repeatn((), NUM_UIDS).map(|_| uid()).collect();

        uids.sort_unstable();
        uids.dedup();
        assert_eq!(uids.len(), NUM_UIDS);
    }
}
