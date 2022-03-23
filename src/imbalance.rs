use crate::PartId;
use itertools::Itertools;
use num::FromPrimitive;
use num::ToPrimitive;
use num::Zero;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Sub;

pub fn compute_parts_load<T: Zero + Clone + AddAssign>(
    partition: &[PartId],
    num_parts: PartId,
    weights: impl IntoIterator<Item = T>,
) -> Vec<T> {
    debug_assert!(*partition.iter().max().unwrap_or(&0) < num_parts);
    partition
        .iter()
        .zip(weights)
        .fold(vec![T::zero(); num_parts], |mut acc, (&part, w)| {
            acc[part] += w;
            acc
        })
}

/// Compute the imbalance of the given partition.
pub fn imbalance<T>(num_parts: usize, partition: &[PartId], weights: &[T]) -> f64
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
    let total_weight: T = weights.iter().cloned().sum();
    if total_weight.is_zero() || weights.is_empty() {
        T::zero().to_f64().unwrap()
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

pub fn imbalance_target<T: Zero + Sum + Clone + AddAssign + Sub<Output = T> + PartialOrd + Copy>(
    targets: &[T],
    partition: &[PartId],
    weights: impl IntoIterator<Item = T>,
) -> T {
    let num_parts = targets.len();
    debug_assert!(*partition.iter().max().unwrap_or(&0) < num_parts);
    compute_parts_load(partition, num_parts, weights)
        .iter()
        .zip(targets)
        .map(|(x, t)| *x - *t)
        .minmax() // Use `itertools.minmax()` as it works with PartialOrd
        .into_option()
        .unwrap_or((T::zero(), T::zero()))
        .1
}

pub fn max_imbalance<T: Zero + Clone + Copy + AddAssign + Sum + PartialOrd + Sub<Output = T>>(
    num_parts: usize,
    partition: &[PartId],
    weights: impl IntoIterator<Item = T>,
) -> T {
    compute_parts_load(partition, num_parts, weights)
        .iter()
        .minmax()
        .into_option()
        .map_or_else(T::zero, |m| *m.1 - *m.0)
}
