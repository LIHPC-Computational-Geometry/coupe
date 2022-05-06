use itertools::Itertools;
use num::FromPrimitive;
use num::ToPrimitive;
use num::Zero;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Sub;

pub fn compute_parts_load<W>(partition: &[usize], num_parts: usize, weights: W) -> Vec<W::Item>
where
    W: IntoParallelIterator,
    W::Iter: IndexedParallelIterator,
    W::Item: Zero + Clone + AddAssign,
{
    debug_assert!(*partition.par_iter().max().unwrap_or(&0) < num_parts);

    partition
        .par_iter()
        .zip(weights)
        .fold(
            || vec![W::Item::zero(); num_parts],
            |mut acc, (&part, w)| {
                acc[part] += w;
                acc
            },
        )
        .reduce_with(|mut weights0, weights1| {
            for (w0, w1) in weights0.iter_mut().zip(weights1) {
                *w0 += w1;
            }
            weights0
        })
        .unwrap_or_else(|| vec![W::Item::zero(); num_parts])
}

/// Compute the imbalance of the given partition.
pub fn imbalance<T>(num_parts: usize, partition: &[usize], weights: &[T]) -> f64
where
    T: Clone
        + FromPrimitive
        + ToPrimitive
        + PartialOrd
        + Zero
        + PartialEq
        + Div<Output = T>
        + Sub<Output = T>
        + Sum,
{
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert!(*partition.par_iter().max().unwrap_or(&0) < num_parts);

    let total_weight: T = weights.iter().cloned().sum();
    if total_weight.is_zero() || weights.is_empty() || num_parts == 0 {
        0.0
    } else {
        let ideal_part_weight = total_weight.to_f64().unwrap() / num_parts.to_f64().unwrap();
        (0..num_parts)
            .map(|part| {
                let part_weight: T = partition
                    .iter()
                    .zip(weights)
                    .filter(|(weight_part, _weight)| **weight_part == part)
                    .map(|(_weight_part, weight)| weight.clone())
                    .sum();
                let part_weight: f64 = part_weight.to_f64().unwrap();
                (part_weight - ideal_part_weight) / ideal_part_weight
            })
            .minmax()
            .into_option()
            .unwrap()
            .1
    }
}

pub fn imbalance_target<W>(targets: &[W::Item], partition: &[usize], weights: W) -> W::Item
where
    W: IntoParallelIterator,
    W::Iter: IndexedParallelIterator,
    W::Item: Zero + Sum + Copy + AddAssign + Sub<Output = W::Item> + PartialOrd,
{
    let num_parts = targets.len();
    compute_parts_load(partition, num_parts, weights)
        .iter()
        .zip(targets)
        .map(|(x, t)| *x - *t)
        .max_by(|imb0, imb1| W::Item::partial_cmp(imb0, imb1).unwrap())
        .unwrap_or_else(W::Item::zero)
}

pub fn max_imbalance<W>(num_parts: usize, partition: &[usize], weights: W) -> W::Item
where
    W: IntoParallelIterator,
    W::Iter: IndexedParallelIterator,
    W::Item: Zero + Sum + Copy + AddAssign + Sub<Output = W::Item> + PartialOrd,
{
    compute_parts_load(partition, num_parts, weights)
        .iter()
        .minmax()
        .into_option()
        .map_or_else(W::Item::zero, |m| *m.1 - *m.0)
}
