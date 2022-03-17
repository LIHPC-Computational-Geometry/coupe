use num::Zero;
use std::ops::AddAssign;

/// Implementation of the greedy algorithm.
fn greedy<T: Ord + Zero + Clone + AddAssign>(
    partition: &mut [usize],
    weights: impl IntoIterator<Item = T>,
    num_parts: usize,
) {
    if num_parts < 2 {
        return;
    }

    // Initialization: make the partition and record the weight of each part in another vector.
    let mut weights: Vec<_> = weights
        .into_iter()
        .zip(0..) // Keep track of the weights' indicies
        .collect();
    assert_eq!(partition.len(), weights.len());
    weights.sort_unstable();
    let mut part_weights = vec![T::zero(); num_parts];

    // Put each weight in the lightweightest part.
    for (weight, weight_id) in weights.into_iter().rev() {
        let (min_part_weight_idx, _min_part_weight) = part_weights
            .iter()
            .enumerate()
            .min_by_key(|(_idx, part_weight)| *part_weight)
            .unwrap();
        partition[weight_id] = min_part_weight_idx;
        part_weights[min_part_weight_idx] += weight;
    }
}

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
pub struct Greedy {
    pub part_count: usize,
}

impl<W> crate::Partition<W> for Greedy
where
    W: IntoIterator,
    W::Item: Ord + Zero + Clone + AddAssign,
{
    type Metadata = ();
    type Error = std::convert::Infallible;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        weights: W,
    ) -> Result<Self::Metadata, Self::Error> {
        greedy(part_ids, weights, self.part_count);
        Ok(())
    }
}
