/// Split `total_work` among a given number of threads.
///
/// Returns `(thread_count, work_per_thread)`, where `thread_count` is the
/// amount of threads that have actual work, and `work_per_thread` is the
/// maximum amount of work these thread have (RCB here splits by chunks, thus
/// the last thread will not have this much work).
///
/// # Panics
///
/// Panics if either argument is zero.
pub fn work_share(total_work: usize, max_threads: usize) -> (usize, usize) {
    let max_threads = usize::min(total_work, max_threads);

    // ceil(total_work / max_threads)
    let work_per_thread = total_work.div_ceil(max_threads);

    // ceil(total_work / work_per_thread)
    let thread_count = total_work.div_ceil(work_per_thread);

    (work_per_thread, thread_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_share() {
        assert_eq!(work_share(100, 4), (25, 4));
        assert_eq!(work_share(101, 4), (26, 4));

        assert_eq!(work_share(100, 20), (5, 20));
        assert_eq!(work_share(101, 20), (6, 17));

        assert_eq!(work_share(100, 100), (1, 100));
        assert_eq!(work_share(100, 101), (1, 100));

        assert_eq!(work_share(100, 1), (100, 1));
        assert_eq!(work_share(100, 2), (50, 2));
        assert_eq!(work_share(100, 3), (34, 3));
        assert_eq!(work_share(1, 100), (1, 1));
        assert_eq!(work_share(2, 100), (1, 2));
        assert_eq!(work_share(3, 100), (1, 3));
    }
}
