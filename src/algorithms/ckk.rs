use num::FromPrimitive;
use num::ToPrimitive;
use std::iter::Sum;
use std::ops::Add;
use std::ops::Sub;

/// Adds an element `e` to a vector `v` and maintain order.
fn add<T: Ord>(v: &mut Vec<T>, e: T) -> usize {
    match v.binary_search(&e) {
        Ok(index) | Err(index) => {
            v.insert(index, e);
            index
        }
    }
}

/// Type stored in each iteration of the algorithm to allow backtracking.
struct Step {
    /// The highest value picked by the algorithm at this step.
    a: usize,

    /// The second highest value picked by the algorithm at this step.
    b: usize,

    /// Whether `a` and `b` must end up in separate parts or not.
    separate: bool,
}

fn ckk_bipart_build(partition: &mut [usize], last_weight: usize, steps: &[Step]) {
    partition[last_weight] = 0;
    for Step { a, b, separate } in steps.iter().rev() {
        if *separate {
            partition[*b] = 1 - partition[*a];
        } else {
            partition[*b] = partition[*a];
        }
    }
}

fn ckk_bipart_rec<T>(
    partition: &mut [usize],
    weights: &mut Vec<(T, usize)>,
    tolerance: T,
    steps: &mut Vec<Step>,
) -> bool
where
    T: Ord + Add<Output = T> + Sub<Output = T> + Copy,
{
    debug_assert_ne!(weights.len(), 0);

    if weights.len() == 1 {
        let (last_weight, last_id) = weights[0];
        if last_weight <= tolerance {
            ckk_bipart_build(partition, last_id, steps);
            return true;
        }
        return false;
    }

    let (a_weight, a_id) = weights.pop().unwrap();
    let (b_weight, b_id) = weights.pop().unwrap();

    let a_minus_b = (a_weight - b_weight, a_id);
    let a_minus_b_idx = add(weights, a_minus_b);
    steps.push(Step {
        a: a_id,
        b: b_id,
        separate: true,
    });
    if ckk_bipart_rec(partition, weights, tolerance, steps) {
        return true;
    }

    weights.remove(a_minus_b_idx);
    steps.pop();
    let a_plus_b = (a_weight + b_weight, a_id);
    let a_plus_b_idx = add(weights, a_plus_b);
    steps.push(Step {
        a: a_id,
        b: b_id,
        separate: true,
    });
    if ckk_bipart_rec(partition, weights, tolerance, steps) {
        return true;
    }

    weights.remove(a_plus_b_idx);
    steps.pop();
    weights.push((b_weight, b_id));
    weights.push((a_weight, a_id));

    false
}

pub fn ckk_bipart<I, T>(partition: &mut [usize], weights: I, tolerance: f64) -> bool
where
    I: IntoIterator<Item = T>,
    T: Sum + Add<Output = T> + Sub<Output = T>,
    T: FromPrimitive + ToPrimitive,
    T: Ord + Default + Copy,
{
    let mut weights: Vec<(T, usize)> = weights.into_iter().zip(0..).collect();
    debug_assert_eq!(weights.len(), partition.len());
    if weights.is_empty() {
        return true;
    }
    weights.sort_unstable();

    let sum: T = weights.iter().map(|(weight, _idx)| *weight).sum();
    let tolerance = T::from_f64(sum.to_f64().unwrap() * tolerance).unwrap();

    let mut steps = Vec::new();

    ckk_bipart_rec(partition, &mut weights, tolerance, &mut steps)
}

// TODO tests
