use num::Zero;
use std::ops::AddAssign;

/// Implementation of the greedy algorithm.
pub fn greedy<T: Ord + Zero + Clone + AddAssign>(
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
