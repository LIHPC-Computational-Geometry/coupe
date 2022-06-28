use super::Error;
use std::iter::Sum;
use std::ops::Add;
use std::ops::Sub;

/// Adds an element `e` to a vector `v` and maintain order.
fn add<T: PartialOrd>(v: &mut Vec<T>, e0: T) -> usize {
    match v.binary_search_by(|e| crate::partial_cmp(e, &e0)) {
        Ok(index) | Err(index) => {
            v.insert(index, e0);
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

#[allow(clippy::ptr_arg)] // clippy bug
fn ckk_bipart_rec<T>(
    partition: &mut [usize],
    weights: &mut Vec<(T, usize)>,
    tolerance: T,
    steps: &mut Vec<Step>,
) -> bool
where
    T: CkkWeight,
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

fn ckk_bipart<I, T>(partition: &mut [usize], weights: I, tolerance: f64) -> Result<(), Error>
where
    I: IntoIterator<Item = T>,
    T: CkkWeight,
{
    let mut weights: Vec<(T, usize)> = weights.into_iter().zip(0..).collect();
    if weights.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: weights.len(),
        });
    }
    if weights.is_empty() {
        return Ok(());
    }
    weights.sort_unstable_by(crate::partial_cmp);

    let sum: T = weights.iter().map(|(weight, _idx)| *weight).sum();
    let tolerance = T::from_f64(sum.to_f64().unwrap() * tolerance).unwrap();

    let mut steps = Vec::new();

    if ckk_bipart_rec(partition, &mut weights, tolerance, &mut steps) {
        Ok(())
    } else {
        Err(Error::NotFound)
    }
}

/// # Complete Karmarkar-Karp algorithm
///
/// Extension of the
/// [Karmarkar-Karp number partitioning algorithm][crate::KarmarkarKarp] that
/// explores all possible solutions until the `tolerance` constraint is
/// respected.
///
/// This algorithm is currently implemented in the bi-partitioning case only.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), coupe::Error> {
/// use coupe::Partition as _;
///
/// let weights: [i32; 4] = [3, 5, 3, 9];
/// let mut partition = [0; 4];
///
/// coupe::CompleteKarmarkarKarp { tolerance: 0.1 }
///     .partition(&mut partition, weights)?;
/// # Ok(())
/// # }
/// ```
///
/// # Reference
///
/// Korf, Richard E., 1998. A complete anytime algorithm for number
/// partitioning. *Artificial Intelligence*, 106(2):181 – 203.
/// <doi:10.1016/S0004-3702(98)00086-1>.
#[derive(Clone, Copy, Debug)]
pub struct CompleteKarmarkarKarp {
    /// Constraint on the normalized imbalance between the two parts.
    pub tolerance: f64,
}

/// Trait alias for values accepted as weights by [CompleteKarmarkarKarp].
pub trait CkkWeight
where
    Self: Copy + Sum + PartialOrd + num::FromPrimitive + num::ToPrimitive,
    Self: Add<Output = Self> + Sub<Output = Self>,
{
}

impl<T> CkkWeight for T
where
    Self: Copy + Sum + PartialOrd + num::FromPrimitive + num::ToPrimitive,
    Self: Add<Output = Self> + Sub<Output = Self>,
{
}

impl<W> crate::Partition<W> for CompleteKarmarkarKarp
where
    W: IntoIterator,
    W::Item: CkkWeight,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        weights: W,
    ) -> Result<Self::Metadata, Self::Error> {
        ckk_bipart(part_ids, weights, self.tolerance)
    }
}

// TODO tests
