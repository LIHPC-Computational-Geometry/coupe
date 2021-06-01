use crate::imbalance::compute_parts_load;
use crate::RunInfo;
use itertools::Itertools;
use num::One;
use num::Zero;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;

pub fn vn_best_mono<T>(partition: &mut [usize], criterion: &[T], nb_parts: usize) -> RunInfo
where
    T: AddAssign + Sub<Output = T> + Div<Output = T> + Mul<Output = T>,
    for<'a> &'a T: Sub<Output = T>,
    T: Zero + One,
    T: Ord + Copy,
{
    let two = {
        let mut two = T::one();
        two += T::one();
        two
    };

    assert_eq!(partition.len(), criterion.len());

    // We expect weights to be non-negative values
    assert!(criterion.iter().all(|weight| *weight >= T::zero()));
    // We check if all weights are 0 because this screw the gain table
    // initialization and a quick fix could interfere with the algorithm.
    if partition.is_empty()
        || criterion.is_empty()
        || criterion.iter().all(|item| item.is_zero())
        || nb_parts < 2
    {
        return RunInfo::skip();
    }

    let mut part_loads = compute_parts_load(partition, nb_parts, criterion);
    let mut criterion: Vec<(T, usize)> = criterion.iter().copied().zip(0..).collect();
    criterion.sort_unstable();

    let mut algo_iterations = 0;

    // The partition is optimized as long as the best move induces a gain higher than 0.
    loop {
        let ((underweight_part, _), (overweight_part, _)) = part_loads
            .iter()
            .enumerate()
            .minmax_by_key(|&(_part, load)| load)
            .into_option()
            .unwrap();
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

    RunInfo {
        algo_iterations: Some(algo_iterations),
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
        let mut part = vec![0 as usize; w.len()];
        let imb_ini = imbalance::imbalance(2, &part, &w);
        vn_best_mono(&mut part, &w, 2);
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
                    (prop::collection::vec(0..1000_000u64, num_weights),
                        prop::collection::vec(0..1usize, num_weights))
                })
        ) {
            let imb_ini = imbalance::max_imbalance(2, &partition, &weights);
            vn_best_mono(&mut partition, &weights, 2);
            let imb_end = imbalance::max_imbalance(2, &partition, &weights);
            // Not sure if it is true for max_imbalance (i.e. weighter - lighter)
            proptest::prop_assert!(imb_end <= imb_ini, "{} < {}", imb_ini, imb_end);
        }
    );
}
