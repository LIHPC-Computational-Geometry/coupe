use super::Error;
use std::ops::AddAssign;

/// Implementation of the greedy algorithm.
fn greedy<T>(
    partition: &mut [usize],
    weights: impl IntoIterator<Item = T>,
    part_count: usize,
) -> Result<(), Error>
where
    T: GreedyWeight,
{
    if part_count < 2 {
        partition.fill(0);
        return Ok(());
    }

    // Initialization: make the partition and record the weight of each part in another vector.
    let mut weights: Vec<_> = weights
        .into_iter()
        .zip(0..) // Keep track of the weights' indicies
        .collect();

    if weights.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: weights.len(),
        });
    }

    weights.sort_unstable_by(crate::partial_cmp);
    let mut part_weights = vec![T::zero(); part_count];

    // Put each weight in the lightweightest part.
    for (weight, weight_id) in weights.into_iter().rev() {
        let (min_part_weight_idx, _min_part_weight) = part_weights
            .iter()
            .enumerate()
            .min_by(|(_, part_weight0), (_, part_weight1)| {
                crate::partial_cmp(part_weight0, part_weight1)
            })
            .unwrap(); // Will not panic because !part_weights.is_empty()
        partition[weight_id] = min_part_weight_idx;
        part_weights[min_part_weight_idx] += weight;
    }

    Ok(())
}

/// Trait alias for values accepted as weights by [Greedy].
pub trait GreedyWeight
where
    Self: PartialOrd + num::Zero + Clone + AddAssign,
{
}

impl<T> GreedyWeight for T where Self: PartialOrd + num::Zero + Clone + AddAssign {}

/// # Greedy number partitioning algorithm
///
/// Greedily assign weights to each part.
///
/// # Example
///
/// ```rust
/// use coupe::Partition as _;
/// use coupe::Real;
///
/// let weights = [3.2, 6.8, 10.0, 7.5].map(Real::from);
/// let mut partition = [0; 4];
///
/// coupe::Greedy { part_count: 2 }
///     .partition(&mut partition, weights)
///     .unwrap();
/// ```
///
/// # Reference
///
/// Horowitz, Ellis and Sahni, Sartaj, 1974. Computing partitions with
/// applications to the knapsack problem. *J. ACM*, 21(2):277–292.
/// <doi:10.1145/321812.321823>.
#[derive(Debug)]
pub struct Greedy {
    pub part_count: usize,
}

impl<W> crate::Partition<W> for Greedy
where
    W: IntoIterator,
    W::Item: GreedyWeight,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        weights: W,
    ) -> Result<Self::Metadata, Self::Error> {
        greedy(part_ids, weights, self.part_count)
    }
}
