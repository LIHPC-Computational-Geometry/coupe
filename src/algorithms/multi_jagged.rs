//! An implementation of the Multi-Jagged spatial partitioning
//! inpired by "Multi-Jagged: A Scalable Parallel Spatial Partitioning Algorithm"
//! by Mehmet Deveci, Sivasankaran Rajamanickam, Karen D. Devine, Umit V. Catalyurek
//!
//! It improves over RCB by following the same idea but by creating more than two subparts
//! in each iteration which leads to decreasing recursion depth.

use crate::geometry::*;
use itertools::iproduct;
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;
use std::sync::atomic::{self, AtomicPtr};
use std::sync::Arc;

fn is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    let p: u32 = (n as f64).sqrt() as u32;

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

// Computes from a set of points, how many sections will be made at each iteration;
fn partition_scheme(_points: &[Point2D], num_parts: usize) -> Vec<usize> {
    // for now the points are ignored
    // TODO: improve by adapting scheme with geometry, e.g. aspect ratio
    let primes = prime_factors(num_parts as u32);

    primes.into_iter().map(|p| p as usize).collect()
}

// returns split indices
fn split_weights(weights: &[f64], num_parts: usize) -> Vec<usize> {
    let target_weight = weights.iter().sum::<f64>() / num_parts as f64;
    let mut ret = vec![];
    let mut idx = 0;
    for _ in 0..num_parts {
        let mut weight = target_weight;
        loop {
            weight -= weights[idx];

            idx += 1;
        }
    }

    ret
}

pub fn multi_jagged_2d_with_scheme(
    points: &[Point2D],
    weights: &[f64],
    partition_scheme: Vec<usize>,
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
        Arc::new(AtomicPtr::new(initial_partition.as_mut_ptr())),
        true,
        &partition_scheme,
    );

    initial_partition
}

fn multi_jagged_2d_recurse(
    points: &[Point2D],
    weights: &[f64],
    permutation: &mut [usize],
    partition: Arc<AtomicPtr<ProcessUniqueId>>,
    x_axis: bool,
    partition_scheme: &[usize],
) {
    if let Some(num_splits) = partition_scheme.iter().next() {
        // recurse
        axis_sort(points, permutation, x_axis);

        let split_positions = compute_split_positions(weights, permutation, *num_splits);
        let mut sub_permutations = split_at_mut_many(permutation, &split_positions);

        let x_axis = !x_axis;
        sub_permutations.par_iter_mut().for_each(|permu| {
            multi_jagged_2d_recurse(
                points,
                weights,
                permu,
                partition.clone(),
                x_axis,
                &partition_scheme[1..],
            )
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
) -> Vec<usize> {
    unimplemented!()
}

fn split_at_mut_many<'a, T>(slice: &'a mut [T], positions: &[usize]) -> Vec<&'a mut [T]> {
    unimplemented!()
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
    fn test_prime_factors() {
        assert_eq!(
            prime_factors(2 * 3 * 3 * 5 * 7 * 11 * 13 * 17),
            vec![2, 3, 3, 5, 7, 11, 13, 17]
        );
    }
}
