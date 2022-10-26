use rayon::iter::IndexedParallelIterator;

mod fold_chunks;

pub use self::fold_chunks::FoldChunks;

pub trait IndexedParallelIteratorExt: IndexedParallelIterator {
    /// Splits an iterator into fixed-size chunks, performing a sequential [`fold()`] on
    /// each chunk.
    ///
    /// Returns an iterator that produces a folded result for each chunk of items
    /// produced by this iterator.
    ///
    /// This works essentially like:
    ///
    /// ```text
    /// iter.chunks(chunk_size)
    ///     .map(|chunk|
    ///         chunk.into_iter()
    ///             .fold(identity, fold_op)
    ///     )
    /// ```
    ///
    /// except there is no per-chunk allocation overhead.
    ///
    /// [`fold()`]: std::iter::Iterator#method.fold
    ///
    /// **Panics** if `chunk_size` is 0.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rayon::prelude::*;
    /// let nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let chunk_sums = nums.into_par_iter().fold_chunks(2, || 0, |a, n| a + n).collect::<Vec<_>>();
    /// assert_eq!(chunk_sums, vec![3, 7, 11, 15, 19]);
    /// ```
    #[track_caller]
    fn fold_chunks<ID, F, U>(
        self,
        chunk_size: usize,
        identity: ID,
        fold_op: F,
    ) -> FoldChunks<Self, ID, F>
    where
        ID: Fn() -> U + Send + Sync,
        F: Fn(U, Self::Item) -> U + Send + Sync,
    {
        assert!(chunk_size != 0, "chunk_size must not be zero");
        FoldChunks::new(self, chunk_size, identity, fold_op)
    }
}

impl<I> IndexedParallelIteratorExt for I where I: IndexedParallelIterator {}

/// Divide `n` by `divisor`, and round up to the nearest integer
/// if not evenly divisible.
#[inline]
fn div_round_up(n: usize, divisor: usize) -> usize {
    debug_assert!(divisor != 0, "Division by zero!");
    if n == 0 {
        0
    } else {
        (n - 1) / divisor + 1
    }
}
