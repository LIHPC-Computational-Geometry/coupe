use crate::imbalance::compute_parts_load;
use crate::Error;
use itertools::Itertools;
use num::One;
use num::Zero;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;

fn vn_best_mono<W, T>(
    partition: &mut [usize],
    criterion: W,
    nb_parts: usize,
) -> Result<usize, Error>
where
    W: IntoIterator<Item = T>,
    T: AddAssign + Sub<Output = T> + Div<Output = T> + Mul<Output = T>,
    T: Zero + One,
    T: Ord + Copy,
{
    let two = {
        let mut two = T::one();
        two += T::one();
        two
    };

    let mut criterion: Vec<(T, usize)> = criterion.into_iter().zip(0..).collect();

    if criterion.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: criterion.len(),
        });
    }
    if criterion
        .iter()
        .any(|(weight, _weight_id)| *weight < T::zero())
    {
        return Err(Error::NegativeValues);
    }

    // We check if all weights are 0 because this screw the gain table
    // initialization and a quick fix could interfere with the algorithm.
    if partition.is_empty()
        || criterion.is_empty()
        || criterion
            .iter()
            .all(|(weight, _weight_id)| weight.is_zero())
        || nb_parts < 2
    {
        return Ok(0);
    }

    let mut part_loads = compute_parts_load(
        partition,
        nb_parts,
        criterion.iter().map(|(weight, _weight_id)| *weight),
    );
    criterion.sort_unstable();

    let mut algo_iterations = 0;

    // The partition is optimized as long as the best move induces a gain higher than 0.
    loop {
        let ((underweight_part, _), (overweight_part, _)) = part_loads
            .iter()
            .enumerate()
            .minmax_by_key(|&(_part, load)| load)
            .into_option()
            .unwrap(); // Won't panic because part_loads as at least two elements.
        let imbalance = part_loads[overweight_part] - part_loads[underweight_part];

        let maybe_nearest = (|| -> Option<usize> {
            let target = imbalance / two;
            let mut above: Option<usize>;
            let mut below: Option<usize>;
            match criterion.binary_search(&(target, 0)) {
                Ok(target_idx) => {
                    if partition[criterion[target_idx].1] == overweight_part {
                        return Some(target_idx);
                    }
                    above = Some(target_idx);
                    below = target_idx.checked_sub(1);
                }
                Err(above_idx) => {
                    above = (above_idx < criterion.len()).then(|| above_idx);
                    below = above_idx.checked_sub(1);
                }
            }
            loop {
                let (chosen, is_above) = if let Some(above) = above {
                    if let Some(below) = below {
                        if criterion[above].0 - target < target - criterion[below].0 {
                            (above, true)
                        } else {
                            (below, false)
                        }
                    } else {
                        (above, true)
                    }
                } else if let Some(below) = below {
                    (below, false)
                } else {
                    return None;
                };
                if partition[criterion[chosen].1] == overweight_part {
                    return Some(chosen);
                }
                if is_above {
                    above = above.map(|i| i + 1).filter(|i| *i < criterion.len());
                } else {
                    below = below.and_then(|i| i.checked_sub(1));
                }
            }
        })();
        let (nearest_weight, id) = match maybe_nearest {
            Some(idx) => criterion[idx],
            None => break,
        };
        if imbalance <= nearest_weight || nearest_weight.is_zero() {
            // The move would either not change the imbalance, or increase it.
            break;
        }
        partition[id] = underweight_part;
        part_loads[overweight_part] = part_loads[overweight_part] - nearest_weight;
        part_loads[underweight_part] += nearest_weight;

        algo_iterations += 1;
    }

    Ok(algo_iterations)
}

/// # Steepest descent Vector-of-Numbers algorithm
///
/// This algorithm greedily moves weights from parts to parts in such a way that
/// the balance gain is maximized on each step.
///
/// # Example
///
/// ```rust
/// use coupe::Partition as _;
/// use rand;
///
/// let part_count = 2;
/// let mut partition = [0; 4];
/// let weights = [4, 6, 2, 9];
///
/// coupe::Random { rng: rand::thread_rng(), part_count }
///     .partition(&mut partition, ())
///     .unwrap();
/// coupe::VnBest { part_count }
///     .partition(&mut partition, weights)
///     .unwrap();
/// ```
///
/// # Reference
///
/// Remi Barat. Load Balancing of Multi-physics Simulation by Multi-criteria
/// Graph Partitioning. Other [cs.OH]. Université de Bordeaux, 2017. English.
/// NNT : 2017BORD0961. tel-01713977
pub struct VnBest {
    pub part_count: usize,
}

impl<W> crate::Partition<W> for VnBest
where
    W: IntoIterator,
    W::Item: AddAssign + Sub<Output = W::Item> + Div<Output = W::Item> + Mul<Output = W::Item>,
    W::Item: Zero + One,
    W::Item: Ord + Copy,
{
    type Metadata = usize;
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        weights: W,
    ) -> Result<Self::Metadata, Self::Error> {
        if self.part_count < 2 {
            return Ok(0);
        }
        vn_best_mono(part_ids, weights, self.part_count)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::*;

    #[test]
    fn small() {
        let w = [1, 2, 3, 4, 5, 6];
        let mut part = vec![0; w.len()];
        let imb_ini = imbalance::imbalance(2, &part, &w);
        vn_best_mono(&mut part, w, 2).unwrap();
        let imb_end = imbalance::imbalance(2, &part, &w);
        assert!(imb_end < imb_ini);
        println!("imbalance : {} < {}", imb_end, imb_ini);
    }

    proptest!(
        #![proptest_config(ProptestConfig{timeout: 2000, ..ProptestConfig::default()})]

        /// vn_best should always improve balance!
        #[test]
        fn improve(
            (weights, mut partition) in
                (2..2000usize).prop_flat_map(|num_weights| {
                    (prop::collection::vec(0..1_000_000_u64, num_weights),
                        prop::collection::vec(0..1usize, num_weights))
                })
        ) {
            let imb_ini = imbalance::max_imbalance(2, &partition, weights.iter().cloned());
            vn_best_mono::<_, u64>(&mut partition, weights.iter().cloned(), 2)
                .unwrap();
            let imb_end = imbalance::max_imbalance(2, &partition, weights.iter().cloned());
            // Not sure if it is true for max_imbalance (i.e. weighter - lighter)
            proptest::prop_assert!(imb_end <= imb_ini, "{} < {}", imb_ini, imb_end);
        }
    );
}
