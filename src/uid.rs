use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

static ID_COUNTER: AtomicU32 = AtomicU32::new(0);

pub fn uid() -> crate::PartId {
    ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uid_are_unique() {
        use rayon::iter::ParallelIterator;

        const NUM_UIDS: usize = 10000;

        let mut uids: Vec<_> = rayon::iter::repeatn((), NUM_UIDS).map(|_| uid()).collect();

        uids.sort_unstable();
        uids.dedup();
        assert_eq!(uids.len(), NUM_UIDS);
    }
}
