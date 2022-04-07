use super::Error;
use std::collections::BinaryHeap;
use std::ops::Sub;
use std::ops::SubAssign;

use num::Zero;

/// Implementation of the Karmarkar-Karp algorithm (bi-partitioning case).
///
/// # Differences with the k-partitioning implementation
///
/// This function has better performance than [kk] called with `num_parts == 2`.
fn kk_bipart<T>(partition: &mut [usize], weights: impl Iterator<Item = T>)
where
    T: Ord + Sub<Output = T>,
{
    let mut weights: BinaryHeap<(T, usize)> = weights
        .into_iter()
        .zip(0..) // Keep track of the weights' indicies
        .collect();

    // Core algorithm: find the imbalance of the partition.
    // "opposites" is built in this loop to backtrack the solution. It tracks weights that must end
    // up in opposite parts.

    let mut opposites = Vec::with_capacity(weights.len());
    while 2 <= weights.len() {
        let (a_weight, a_id) = weights.pop().unwrap();
        let (b_weight, b_id) = weights.pop().unwrap();

        opposites.push((a_id, b_id));

        // put "a-b" in the same part as "a".
        weights.push((a_weight - b_weight, a_id));
    }

    // Backtracking.
    // We use an array that maps weight IDs to their part (true or false) and their weight value.
    // It is initialized with the last element of "weights" (which is the imbalance of the
    // partition).

    let (_imbalance, last_diff) = weights.pop().unwrap();
    partition[last_diff] = 0;
    for (a, b) in opposites.into_iter().rev() {
        // put "b" in the opposite part of "a" (which is were "a-b" was put).
        partition[b] = 1 - partition[a];
    }
}

/// Implementation of the Karmarkar-Karp algorithm (general case).
fn kk<T, I>(partition: &mut [usize], weights: I, num_parts: usize)
where
    T: Zero + Ord + Sub<Output = T> + SubAssign + Copy,
    I: Iterator<Item = T> + ExactSizeIterator,
{
    // Initialize "m", a "k*num_weights" matrix whose first column is "weights".
    let weight_count = weights.len();
    let mut m: BinaryHeap<Vec<(T, usize)>> = weights
        .zip(0..)
        .map(|(w, id)| {
            let mut v: Vec<(T, usize)> = (0..num_parts)
                .map(|p| (T::zero(), weight_count * p + id))
                .collect();
            v[0].0 = w;
            v
        })
        .collect();

    // Core algorithm: same as the bi-partitioning case. However, instead of putting the two
    // largest weights in two different parts, the largest weight of each row is put into the same
    // part as the smallest one, and so on.

    let mut opposites = Vec::with_capacity(weight_count);
    while 2 <= m.len() {
        let a = m.pop().unwrap();
        let b = m.pop().unwrap();

        // tuples = [ (a0, bn), (a1, bn-1), ... ]
        let tuples: Vec<_> = a
            .iter()
            .zip(b.iter().rev())
            .map(|((_, a_id), (_, b_id))| (*a_id, *b_id))
            .collect();

        // e = [ a0 + bn, a1 + bn-1, ... ]
        let mut e: Vec<_> = a
            .iter()
            .zip(b.iter().rev())
            .map(|(a, b)| (a.0 + b.0, a.1))
            .collect();
        e.sort_unstable_by(|ei, ej| T::cmp(&ej.0, &ei.0));

        let emin = e[e.len() - 1].0;
        for ei in &mut e {
            ei.0 -= emin;
        }
        opposites.push(tuples);
        m.push(e);
    }

    // Backtracking. Same as the bi-partitioning case.

    // parts = [ [m0i] for m0i in m[0] ]
    let mut parts: Vec<usize> = vec![0; num_parts * weight_count];
    let imbalance = m.pop().unwrap(); // first and last element of "m".
    for (i, w) in imbalance.into_iter().enumerate() {
        // Put each remaining element in a different part.
        parts[w.1] = i;
    }
    for tuples in opposites.into_iter().rev() {
        for (a, b) in tuples {
            parts[b] = parts[a];
        }
    }

    parts.truncate(partition.len());
    partition.copy_from_slice(&parts);
}

/// # Karmarkar-Karp algorithm
///
/// Also called the Largest Differencing Method.
///
/// Similar to the [greedy number partitioning algorithm][crate::Greedy], but
/// instead of puting the highest weight into the lowest part, it puts the two
/// highest weights in two different parts and keeps their difference.
///
/// # Example
///
/// ```rust
/// use coupe::Partition as _;
///
/// let weights = [3, 5, 3, 9];
/// let mut partition = [0; 4];
///
/// coupe::KarmarkarKarp { part_count: 3 }
///     .partition(&mut partition, weights)
///     .unwrap();
/// ```
///
/// # Reference
///
/// Karmarkar, Narenda and Karp, Richard M., 1983. The differencing method of
/// set partitioning. Technical report, Berkeley, CA, USA.
pub struct KarmarkarKarp {
    pub part_count: usize,
}

impl<W> crate::Partition<W> for KarmarkarKarp
where
    W: IntoIterator,
    W::IntoIter: ExactSizeIterator,
    W::Item: Zero + Ord + Sub<Output = W::Item> + SubAssign + Copy,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        weights: W,
    ) -> Result<Self::Metadata, Self::Error> {
        if self.part_count < 2 || part_ids.len() < 2 {
            return Ok(());
        }
        let weights = weights.into_iter();
        if weights.len() != part_ids.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: weights.len(),
            });
        }
        if self.part_count == 2 {
            // The bi-partitioning is a special case that can be handled faster
            // than the general case.
            kk_bipart(part_ids, weights);
        } else {
            kk(part_ids, weights, self.part_count);
        }
        Ok(())
    }
}
