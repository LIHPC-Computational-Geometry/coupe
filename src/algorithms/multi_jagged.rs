//! An implementation of the Multi-Jagged spatial partitioning
//! inpired by "Multi-Jagged: A Scalable Parallel Spatial Partitioning Algorithm"
//! by Mehmet Deveci, Sivasankaran Rajamanickam, Karen D. Devine, Umit V. Catalyurek
//!
//! It improves over RCB by following the same idea but by creating more than two subparts
//! in each iteration which leads to decreasing recursion depth.

use crate::geometry::*;
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;
use std::sync::atomic::{self, AtomicPtr};

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

// 7 in 3 steps:
// ceil(7^(1/3)) = 2, 7/2 = 3, 7 % 2 = 1 => one cut
// left: ceil(3 ^ (1/3)) = 2 2*1 + 1 => one cut then cut again
// right: ceil((3 + 1) ^ (1/3)) = 2 => 2*2+0 => two cuts

// Computes from a set of points, how many sections will be made at each iteration;
// fn partition_scheme(points: &[Point2D], num_parts: usize, max_iter: usize) -> Vec<usize> {
//     let approx_root = (num_parts as f32).powf(1. / max_iter as f32).ceil() as usize;
//     let rem = num_parts % approx_root;
// }

// Returns the number of splits for the iteration and an array of target_weights modifier to
// take into account an asymetric split scheme
fn partition_scheme_one_step(num_parts: usize, max_iter: usize) -> PartitionScheme {
    let approx_root = (num_parts as f32).powf(1. / max_iter as f32).ceil() as usize;
    let rem = num_parts % approx_root;
    let quotient = num_parts / approx_root;

    let n = quotient as f32;
    let m = n + 1 as f32;
    let M = rem as f32;
    let N = (approx_root - rem) as f32;

    println!("num_parts = {}", num_parts);
    println!("max_iter = {}", max_iter);
    println!("approx = {}", approx_root);
    println!("rem = {}", rem);
    println!("q = {}", quotient);

    println!("n = {}", n);
    println!("m = {}", m);
    println!("N = {}", N);
    println!("M = {}", M);
    let modifiers = (0..rem)
        .map(|_| m / (n * N + m * M))
        .chain((rem..approx_root).map(|_| n / (n * N + m * M)))
        .collect::<Vec<_>>();

    let next = if rem == 0 && max_iter == 0 {
        None
    } else {
        let mut next = Vec::new();
        for _ in 0..rem {
            next.push(partition_scheme_one_step(quotient + 1, max_iter - 1));
        }
        for _ in rem..approx_root {
            next.push(partition_scheme_one_step(quotient, max_iter - 1));
        }
        Some(next)
    };

    PartitionScheme {
        num_splits: approx_root - 1,
        modifiers,
        next,
    }
}

#[derive(Debug)]
pub struct PartitionScheme {
    pub num_splits: usize,
    pub modifiers: Vec<f32>,
    pub next: Option<Vec<PartitionScheme>>,
}

// Computes from a set of points, how many sections will be made at each iteration;
fn partition_scheme(_points: &[Point2D], num_parts: usize) -> Vec<usize> {
    // for now the points are ignored
    // TODO: improve by adapting scheme with geometry, e.g. aspect ratio
    let primes = prime_factors(num_parts as u32);

    primes.into_iter().map(|p| p as usize).collect()
}

pub fn multi_jagged_2d(
    points: &[Point2D],
    weights: &[f64],
    num_parts: usize,
    max_iter: usize,
) -> Vec<ProcessUniqueId> {
    let partition_scheme = partition_scheme_one_step(num_parts, max_iter);
    multi_jagged_2d_with_scheme(points, weights, partition_scheme)
}

pub fn multi_jagged_2d_with_scheme(
    points: &[Point2D],
    weights: &[f64],
    partition_scheme: PartitionScheme,
) -> Vec<ProcessUniqueId> {
    let len = points.len();
    let mut permutation = (0..len).into_par_iter().collect::<Vec<_>>();
    let initial_id = ProcessUniqueId::new();
    let mut initial_partition = rayon::iter::repeat(initial_id)
        .take(len)
        .collect::<Vec<_>>();

    multi_jagged_2d_recurse(
        points,
        weights,
        &mut permutation,
        &AtomicPtr::new(initial_partition.as_mut_ptr()),
        true,
        partition_scheme,
    );

    initial_partition
}

fn multi_jagged_2d_recurse(
    points: &[Point2D],
    weights: &[f64],
    permutation: &mut [usize],
    partition: &AtomicPtr<ProcessUniqueId>,
    x_axis: bool,
    partition_scheme: PartitionScheme,
) {
    if partition_scheme.num_splits != 0 {
        let num_splits = partition_scheme.num_splits;

        axis_sort(points, permutation, x_axis);

        let split_positions = compute_split_positions(
            weights,
            permutation,
            num_splits,
            &partition_scheme.modifiers,
        );
        let mut sub_permutations = split_at_mut_many(permutation, &split_positions);

        let x_axis = !x_axis;
        sub_permutations
            .par_iter_mut()
            .zip(partition_scheme.next.unwrap())
            .for_each(|(permu, scheme)| {
                multi_jagged_2d_recurse(points, weights, permu, partition, x_axis, scheme)
            });
    } else {
        let part_id = ProcessUniqueId::new();
        permutation.par_iter().for_each(|idx| {
            let ptr = partition.load(atomic::Ordering::Relaxed);
            unsafe { std::ptr::write(ptr.add(*idx), part_id) }
        });
    }
}

fn axis_sort(points: &[Point2D], permutation: &mut [usize], x_axis: bool) {
    if x_axis {
        permutation.par_sort_by(|i1, i2| is_less_cmp_f64(points[*i1].x, points[*i2].x));
    } else {
        permutation.par_sort_by(|i1, i2| is_less_cmp_f64(points[*i1].y, points[*i2].y));
    }
}

fn compute_split_positions(
    weights: &[f64],
    permutation: &[usize],
    num_splits: usize,
    modifiers: &[f32],
) -> Vec<usize> {
    let total_weight = permutation.par_iter().map(|idx| weights[*idx]).sum::<f64>();

    let weight_thresholds = (1..=num_splits)
        .map(|n| total_weight * n as f64 / (num_splits + 1) as f64)
        .collect::<Vec<_>>();

    println!("modifiers = {:?}", modifiers);
    println!("n_splits = {}", num_splits);
    let mut modifiers = modifiers.into_iter();
    let mut consumed_weight = total_weight * *modifiers.next().unwrap() as f64;
    let mut weight_thresholds = Vec::with_capacity(num_splits);

    while let Some(modifier) = modifiers.next() {
        weight_thresholds.push(consumed_weight);
        consumed_weight += total_weight * *modifier as f64;
    }

    assert_eq!(weight_thresholds.len(), num_splits);
    println!("thresholds = {:?}", weight_thresholds);

    let mut ret = Vec::with_capacity(num_splits);

    let mut scan = permutation
        .par_iter()
        .enumerate()
        .fold_with((std::usize::MAX, 0.), |(low, acc), (idx, val)| {
            if idx < low {
                (idx, acc + weights[*val])
            } else {
                (low, acc + weights[*val])
            }
        }).collect::<Vec<_>>()
        .into_iter();

    let mut current_weights_sum = 0.;
    let mut current_weights_sums_cache = Vec::with_capacity(num_splits);

    for threshold in weight_thresholds.iter() {
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
            while sum < threshold {
                idx += 1;
                sum += weights[permutation[idx]];
            }
            idx
        }).collect()
}

// Same as slice::split_at_mut but split in a arbitrary number of subslices
// Sequential since `position` should be small
fn split_at_mut_many<'a, T>(slice: &'a mut [T], positions: &[usize]) -> Vec<&'a mut [T]> {
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

fn is_less_cmp_f64(a: f64, b: f64) -> Ordering {
    if a < b {
        Ordering::Less
    } else {
        Ordering::Greater
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
    fn test_partition_scheme() {
        let scheme = partition_scheme_one_step(7, 3);

        eprintln!("{:#?}", scheme);
        assert!(false);
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
