//! An implementation of the Multi-Jagged spatial partitioning
//! inpired by "Multi-Jagged: A Scalable Parallel Spatial Partitioning Algorithm"
//! by Mehmet Deveci, Sivasankaran Rajamanickam, Karen D. Devine, Umit V. Catalyurek
//!
//! It improves over RCB by following the same idea but by creating more than two subparts
//! in each iteration which leads to decreasing recursion depth.

use approx::Ulps;

use crate::geometry::*;
use rayon::prelude::*;

use std::sync::atomic::{self, AtomicPtr};

// prime functions are currently unused but may be useful to compute a
// partitioning scheme based on a prime factor decomposition

#[allow(unused)]
fn is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    let p: u32 = (f64::from(n)).sqrt() as u32;

    for i in 2..=p {
        if n % i == 0 {
            return false;
        }
    }
    true
}

// Computes the list of primes factors of a given number n
#[allow(unused)]
fn prime_factors(mut n: u32) -> Vec<u32> {
    if n <= 1 {
        return vec![1];
    }

    let mut ret = vec![];
    let mut primes = (2..).filter(|n| is_prime(*n));
    let mut current = primes.next().unwrap();
    while n > 1 {
        while n % current == 0 {
            ret.push(current);
            n /= current;
        }
        current = primes.next().unwrap();
    }

    ret
}

#[derive(Debug)]
struct PartitionScheme {
    pub num_splits: usize,
    pub modifiers: Vec<f64>,
    pub next: Option<Vec<PartitionScheme>>,
}

// Computes a partitioning scheme i.e. how to split points at each iteration given
// a number of partitions and a number of max iterations.
//
// The current implementation is a recursive algorithm :
//   - The first split generates num_parts^(1/max_iter) parts.
//   - Some of those parts can be "fat" meaning that they will eventually contain
//     one more part than the "non-fat" one.
//   - Recursion is applied on each part.
fn partition_scheme(num_parts: usize, max_iter: usize) -> PartitionScheme {
    let approx_root = (num_parts as f32).powf(1. / max_iter as f32).ceil() as usize;
    let rem = num_parts % approx_root;
    let quotient = num_parts / approx_root;

    let modifiers = compute_modifiers(approx_root - rem, rem, quotient, quotient + 1);

    let next = if rem == 0 && max_iter == 0 {
        None
    } else {
        let mut next = Vec::new();
        for _ in 0..rem {
            // recurse on "fat" parts
            next.push(partition_scheme(quotient + 1, max_iter - 1));
        }
        for _ in rem..approx_root {
            // recurse on "regular" parts
            next.push(partition_scheme(quotient, max_iter - 1));
        }
        Some(next)
    };

    PartitionScheme {
        num_splits: approx_root - 1,
        modifiers,
        next,
    }
}

// Computes target weight modifiers for irregular partitions.
//
// Example:
//    Consider the unit square [0, 1] x [0, 1] to be a uniform distribution of many points.
//    We want to split it in 5 parts of equal weights with multi-jagged in 2 iterations.
//     - First iteration: we split the square in two parts.
//     - Second iteration: we split the left part in three subparts and the right part
//       in two parts.
//    Or, if we want to split it in 3 parts of equal weight in 2 iterations.
//
//    The first split must be done unevenly and will be done such that after the split,
//    some parts will eventally contain exactly one more final part than others (e.g. below
//    the first left split eventually contains 3 final parts and the right split only contains 2).
//    Note: it also could be possible that the number of final parts difference between parts is higher than 1,
//    but it's not the case here.
//
//    Given an iteration that requires N splits of the current part that contains the total weight W,
//    the modifiers is an array [mod_1, ..., mod_n] of float values. The split should be done such that
//    the i-th part contains the weight mod_i * W.
//
//                  5 parts in 2 iterations                         3 parts in 2 iterations
//               _____________________________                  _____________________________
//    |       | |                    |        | |            | |                  |          |
//    |   1/3 | |                    |        | | 1/2    1/2 | |                  |          |
//    |       | |                    |        | |            | |                  |          |
//    |         |--------------------|        | |            | |                  |          |
//    |       | |                    |        | |            | |                  |          |
//  1 |   1/3 | |                    |--------|                |------------------|          |
//    |       | |                    |        | |            | |                  |          |
//    |       | |                    |        | |            | |                  |          |
//    |         |--------------------|        | | 1/2    1/2 | |                  |          |
//    |       | |                    |        | |            | |                  |          |
//    |   1/3 | |                    |        | |            | |                  |          |
//    |       | |____________________|________| |            | |__________________|__________|
//               ____________________ ________                  __________________ __________
//                         3/5             2/5                            2/3            1/3
//
fn compute_modifiers(
    num_regular_parts: usize,
    num_fat_parts: usize,
    num_regular_subparts: usize, // the total number of subparts in the regular_parts
    num_fat_subparts: usize,     // the total number of subparts in the fat_parts
) -> Vec<f64> {
    let num_subparts = num_regular_parts * num_regular_subparts + num_fat_parts * num_fat_subparts;
    (0..num_fat_parts)
        .map(|_| num_fat_subparts as f64 / num_subparts as f64)
        .chain((0..num_regular_parts).map(|_| num_regular_subparts as f64 / num_subparts as f64))
        .collect()
}

fn multi_jagged<const D: usize>(
    partition: &mut [usize],
    points: &[PointND<D>],
    weights: &[f64],
    num_parts: usize,
    max_iter: usize,
) {
    let partition_scheme = partition_scheme(num_parts, max_iter);
    multi_jagged_with_scheme(partition, points, weights, partition_scheme);
}

fn multi_jagged_with_scheme<const D: usize>(
    partition: &mut [usize],
    points: &[PointND<D>],
    weights: &[f64],
    partition_scheme: PartitionScheme,
) {
    let len = points.len();
    let mut permutation = (0..len).into_par_iter().collect::<Vec<_>>();

    multi_jagged_recurse(
        points,
        weights,
        &mut permutation,
        &AtomicPtr::new(partition.as_mut_ptr()),
        0,
        partition_scheme,
    );
}

fn multi_jagged_recurse<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    permutation: &mut [usize],
    partition: &AtomicPtr<usize>,
    current_coord: usize,
    partition_scheme: PartitionScheme,
) {
    if partition_scheme.num_splits != 0 {
        super::recursive_bisection::axis_sort(points, permutation, current_coord);

        let split_positions =
            compute_split_positions(weights, permutation, &partition_scheme.modifiers);
        let mut sub_permutations = split_at_mut_many(permutation, &split_positions);

        sub_permutations
            .par_iter_mut()
            .zip(partition_scheme.next.unwrap())
            .for_each(|(permu, scheme)| {
                multi_jagged_recurse(
                    points,
                    weights,
                    permu,
                    partition,
                    (current_coord + 1) % D,
                    scheme,
                )
            });
    } else {
        let part_id = crate::uid();
        permutation.par_iter().for_each(|idx| {
            let ptr = partition.load(atomic::Ordering::Relaxed);
            unsafe { std::ptr::write(ptr.add(*idx), part_id) }
        });
    }
}

// This is pub(crate) because it's also used in the hilbert_curve module
pub(crate) fn compute_split_positions(
    weights: &[f64],
    permutation: &[usize],
    modifiers: &[f64],
) -> Vec<usize> {
    let (_last_modifier, modifiers) = modifiers.split_last().unwrap();
    let num_splits = modifiers.len();

    let total_weight = permutation.par_iter().map(|idx| weights[*idx]).sum::<f64>();
    let weight_thresholds: Vec<_> = modifiers
        .iter()
        .scan(0.0, |consumed_weight, modifier| {
            *consumed_weight += total_weight * modifier;
            Some(*consumed_weight)
        })
        .collect();

    let mut scan = permutation
        .par_iter()
        .enumerate()
        .fold_with((std::usize::MAX, 0.), |(low, acc), (idx, val)| {
            (usize::min(idx, low), acc + weights[*val])
        })
        .collect::<Vec<_>>()
        .into_iter();

    let mut ret = Vec::with_capacity(num_splits);
    let mut current_weights_sum = 0.;
    let mut current_weights_sums_cache = Vec::with_capacity(num_splits);

    for threshold in &weight_thresholds {
        // if this condition is verified, it means that a block of the scan contained more than one threshold
        // and the current threshold was skipped during previous iteration. We just
        // push the last element again and skip the rest of the iteration
        if current_weights_sum > *threshold {
            let last = ret[ret.len() - 1];
            ret.push(last);
            let last = current_weights_sums_cache[current_weights_sums_cache.len() - 1];
            current_weights_sums_cache.push(last);
            continue;
        }

        'inner: loop {
            let current = scan.next().unwrap();
            if current_weights_sum + current.1 > *threshold {
                ret.push(current.0);
                current_weights_sums_cache.push(current_weights_sum);
                current_weights_sum += current.1;
                break 'inner;
            }
            current_weights_sum += current.1;
        }
    }

    ret.into_par_iter()
        .zip(current_weights_sums_cache)
        .zip(weight_thresholds)
        .map(|((mut idx, mut sum), threshold)| {
            while sum + weights[permutation[idx]] < threshold
                // multiplication between modifiers and weights can cause nasty
                // rounding precision loss which would put an element in a wrong part
                || Ulps::default().eq(&threshold, &(sum + weights[permutation[idx]]))
            {
                sum += weights[permutation[idx]];
                idx += 1;
            }
            idx
        })
        .collect()
}

// Same as slice::split_at_mut but split in a arbitrary number of subslices
// Sequential since `position` should be small
//
// This is pub(crate) because it's also used in the hilbert_curve module
pub(crate) fn split_at_mut_many<'a, T>(
    slice: &'a mut [T],
    positions: &[usize],
) -> Vec<&'a mut [T]> {
    let ret = Vec::with_capacity(positions.len() + 1);

    let (mut head, tail, _) = positions.iter().fold(
        (ret, slice, 0),
        |(mut acc_ret, acc_slice, drained_count), pos| {
            let (sub, next) = acc_slice.split_at_mut(*pos - drained_count);
            let len = sub.len();
            acc_ret.push(sub);
            (acc_ret, next, drained_count + len)
        },
    );

    head.push(tail);
    head
}

/// # Multi-Jagged algorithm
///
/// This algorithm is inspired by Multi-Jagged: A Scalable Parallel Spatial Partitioning Algorithm"
/// by Mehmet Deveci, Sivasankaran Rajamanickam, Karen D. Devine, Umit V. Catalyurek.
///
/// It improves over [RCB](struct.Rcb.html) by following the same idea but by creating more than two subparts
/// in each iteration which leads to decreasing recursion depth. It also allows to generate a partition
/// of any number of parts.
///
/// More precisely, given a number of parts, the algorithm will generate a "partition scheme", which describes how
/// to perform splits at each iteration, such that the total number of iteration is less than `max_iter`.
///
/// More iteration does not necessarily result in a better partition.
///
/// # Example
///
/// ```rust
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// let points = vec![
///     Point2D::new(0., 0.),
///     Point2D::new(1., 0.),
///     Point2D::new(2., 0.),   
///     Point2D::new(0., 1.),
///     Point2D::new(1., 1.),
///     Point2D::new(2., 1.),
///     Point2D::new(0., 2.),
///     Point2D::new(1., 2.),
///     Point2D::new(2., 2.),
/// ];
/// let weights = [4.2; 9];
/// let mut partition = [0; 9];
///
/// // generate a partition of 4 parts
/// coupe::MultiJagged { part_count: 9, max_iter: 4 }
///     .partition(&mut partition, (&points, &weights))
///     .unwrap();
///
/// for i in 0..9 {
///     for j in 0..9 {
///         if j == i {
///             continue    
///         }
///         assert_ne!(partition[i], partition[j])
///     }
/// }
/// ```
pub struct MultiJagged {
    pub part_count: usize,
    pub max_iter: usize,
}

impl<'a, const D: usize> crate::Partition<(&'a [PointND<D>], &'a [f64])> for MultiJagged {
    type Metadata = ();
    type Error = std::convert::Infallible;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (&'a [PointND<D>], &'a [f64]),
    ) -> Result<Self::Metadata, Self::Error> {
        multi_jagged(part_ids, points, weights, self.part_count, self.max_iter);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(!is_prime(6));
        assert!(is_prime(7));
        assert!(!is_prime(8));
        assert!(!is_prime(9));
        assert!(!is_prime(10));
        assert!(is_prime(11));
        assert!(!is_prime(12));
        assert!(is_prime(13));
    }

    #[test]
    fn test_prime_factors() {
        assert_eq!(
            prime_factors(2 * 3 * 3 * 5 * 7 * 11 * 13 * 17),
            vec![2, 3, 3, 5, 7, 11, 13, 17]
        );
    }

    #[test]
    fn test_split_at_mut_many() {
        let array = &mut [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        let sub_arrays = split_at_mut_many(array, &[1, 3, 6, 9, 11]);

        assert_eq!(
            sub_arrays,
            vec![
                &mut [0][..],
                &mut [1, 2][..],
                &mut [3, 4, 5][..],
                &mut [6, 7, 8][..],
                &mut [9, 10][..],
                &mut [11, 12][..],
            ]
        )
    }
}
