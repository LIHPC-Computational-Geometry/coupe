use crate::Error;
use itertools::Itertools as _;
use num::FromPrimitive;
use num::One;
use num::Zero;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;

fn vn_first_mono<T>(
    partition: &mut [usize],
    weights: &[T],
    num_parts: usize,
) -> Result<usize, Error>
where
    T: VnFirstWeight,
{
    if weights.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: weights.len(),
        });
    }
    assert_ne!(num_parts, 0);
    if weights.is_empty() || num_parts < 2 {
        return Ok(0);
    }

    let mut part_loads =
        crate::imbalance::compute_parts_load(partition, num_parts, weights.par_iter().cloned());
    let total_weight: T = part_loads.iter().cloned().sum();
    if total_weight.is_zero() {
        return Ok(0);
    }

    let (min_load, mut max_load) = part_loads.iter().cloned().minmax().into_option().unwrap();
    let mut imbalance = max_load.clone() - min_load;

    let mut i = weights.len();
    let mut i_last = 0;
    let mut algo_iterations = 0;
    while i != i_last {
        i = (i + 1) % weights.len();

        // loop through the weights.
        let p = partition[i];

        if part_loads[p] < max_load {
            // weight #i is not in the heaviest partition, and thus the move
            // will not reduce the max imbalance.
            continue;
        }

        for q in 0..num_parts {
            // loop through the parts.
            if p == q {
                // weight #i is already in partition #q.
                continue;
            }

            part_loads[p] = part_loads[p].clone() - weights[i].clone();
            part_loads[q] += weights[i].clone();
            let (new_min_load, new_max_load) =
                part_loads.iter().cloned().minmax().into_option().unwrap();
            let new_imbalance = new_max_load.clone() - new_min_load.clone();
            if imbalance < new_imbalance {
                // The move does not decrease the partition imbalance.
                part_loads[p] += weights[i].clone();
                part_loads[q] = part_loads[p].clone() - weights[i].clone();
                continue;
            }
            imbalance = new_imbalance;
            max_load = new_max_load;
            partition[i] = q;
            i_last = i;
        }

        algo_iterations += 1;
    }

    Ok(algo_iterations)
}

#[allow(dead_code)]
pub fn vn_first<const C: usize, T>(
    partition: &mut [usize],
    criteria: &[[T; C]],
    num_parts: usize,
) -> usize
where
    T: AddAssign + SubAssign + Sub<Output = T> + Sum,
    T: Zero + One + FromPrimitive,
    T: Copy + Ord,
{
    if num_parts < 2 || criteria.is_empty() || C == 0 {
        return 0;
    }

    assert_eq!(criteria.len(), partition.len());

    let mut part_loads_per_criterion = {
        let mut loads = vec![[T::zero(); C]; num_parts];
        for (w, weight) in criteria.iter().enumerate() {
            for (part_load, criterion) in loads[partition[w]].iter_mut().zip(weight) {
                *part_load += *criterion;
            }
        }
        loads
    };
    let total_weight_per_criterion = {
        // TODO replace with .collect() once [_; C] implements FromIterator.
        let mut ws = [T::zero(); C];
        for c in 0..C {
            ws[c] = part_loads_per_criterion[c].iter().cloned().sum();
        }
        ws
    };
    if total_weight_per_criterion.contains(&T::zero()) {
        return 0;
    }

    let min_max_loads = |part_loads_per_criterion: &Vec<[T; C]>| -> [(T, T); C] {
        // TODO replace with .collect() once [_; C] implements FromIterator.
        let mut imbs = [(T::zero(), T::zero()); C];
        for c in 0..C {
            imbs[c] = part_loads_per_criterion[c]
                .iter()
                .cloned()
                .minmax()
                .into_option()
                .unwrap();
        }
        imbs
    };

    let (global_min_load, mut global_max_load) = *min_max_loads(&part_loads_per_criterion)
        .iter()
        .max_by_key(|(min_load, max_load)| *max_load - *min_load)
        .unwrap();
    let mut imbalance = global_max_load - global_min_load;

    let mut i = 0;
    let mut i_last = 0;
    let mut algo_iterations = 0;
    while i != i_last {
        i = (i + 1) % criteria.len();

        // loop through the weights.
        let p = partition[i];

        if part_loads_per_criterion[p]
            .iter()
            .all(|criterion_load| *criterion_load < global_max_load)
        {
            // weight #i is not in the heaviest partition, and thus the move
            // will not reduce the max imbalance.
            continue;
        }

        for q in 0..num_parts {
            // loop through the parts.
            if p == q {
                // weight #i is already in partition #q.
                continue;
            }

            for c in 0..C {
                part_loads_per_criterion[p][c] -= criteria[i][c];
                part_loads_per_criterion[q][c] += criteria[i][c];
            }
            let (new_global_min_load, new_global_max_load) =
                *min_max_loads(&part_loads_per_criterion)
                    .iter()
                    .max_by_key(|(min_load, max_load)| *max_load - *min_load)
                    .unwrap();
            let new_imbalance = new_global_max_load - new_global_min_load;
            if imbalance < new_imbalance {
                // The move does not decrease the partition imbalance.
                for c in 0..C {
                    part_loads_per_criterion[p][c] += criteria[i][c];
                    part_loads_per_criterion[q][c] -= criteria[i][c];
                }
                continue;
            }
            imbalance = new_imbalance;
            global_max_load = new_global_max_load;
            partition[i] = q;
            i_last = i;
        }

        algo_iterations += 1;
    }

    algo_iterations
}

/// Trait alias for values accepted as weights by [VnFirst].
pub trait VnFirstWeight
where
    Self: Clone + Send + Sync,
    Self: Sum + PartialOrd + num::FromPrimitive + num::Zero + num::One,
    Self: Sub<Output = Self> + AddAssign,
{
}

impl<T> VnFirstWeight for T
where
    Self: Clone + Send + Sync,
    Self: Sum + PartialOrd + num::FromPrimitive + num::Zero + num::One,
    Self: Sub<Output = Self> + AddAssign,
{
}

/// # Descent Vector-of-Numbers algorithm
///
/// This algorithm moves weights from parts to parts whenever it decreases the
/// imbalance.  See also its [greedy version][crate::VnBest].
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use coupe::Partition as _;
/// use rand;
///
/// let part_count = 2;
/// let mut partition = [0; 4];
/// let weights = [4, 6, 2, 9];
///
/// coupe::Random { rng: rand::thread_rng(), part_count }
///     .partition(&mut partition, ())?;
/// coupe::VnFirst { part_count }
///     .partition(&mut partition, &weights)?;
/// # Ok(())
/// # }
/// ```
///
/// # Reference
///
/// Remi Barat. Load Balancing of Multi-physics Simulation by Multi-criteria
/// Graph Partitioning. Other [cs.OH]. Université de Bordeaux, 2017. English.
/// NNT : 2017BORD0961. tel-01713977
#[derive(Clone, Copy, Debug)]
pub struct VnFirst {
    pub part_count: usize,
}

impl<'a, W> crate::Partition<&'a [W]> for VnFirst
where
    W: VnFirstWeight,
{
    type Metadata = usize;
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        weights: &'a [W],
    ) -> Result<Self::Metadata, Self::Error> {
        vn_first_mono(part_ids, weights, self.part_count)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::*;

    #[test]
    fn small_mono() {
        const W: [i32; 6] = [1, 2, 3, 4, 5, 6];
        let mut part = [0; W.len()];

        vn_first_mono(&mut part, &W, 1).unwrap();
        let imb_ini = imbalance::imbalance(2, &part, &W);
        vn_first_mono(&mut part, &W, 2).unwrap();
        let imb_end = imbalance::imbalance(2, &part, &W);
        assert!(imb_end <= imb_ini);
        println!("imbalance : {} < {}", imb_end, imb_ini);
    }

    proptest!(
        /// vn_first should always improve balance !
        #[test]
        fn improve_mono(
            (weights, mut partition) in
                (2..200_usize).prop_flat_map(|num_weights| {
                    (prop::collection::vec(1..1000_u64, num_weights),
                        prop::collection::vec(0..1_usize, num_weights))
                })
        ) {
            let imb_ini = imbalance::max_imbalance(2, &partition, weights.par_iter().cloned());
            vn_first_mono(&mut partition, &weights, 2).unwrap();
            let imb_end = imbalance::max_imbalance(2, &partition, weights.par_iter().cloned());
            // Not sure if it is true for max_imbalance (i.e. weighter - lighter)
            proptest::prop_assert!(imb_end <= imb_ini);
        }
    );

    #[allow(clippy::needless_collect)] // clippy bug
    #[test]
    fn small() {
        const W: [[i32; 2]; 6] = [[1, 2], [2, 4], [3, 6], [8, 4], [10, 5], [12, 6]];
        let mut part = [0; W.len()];

        vn_first(&mut part, &W, 1);
        let imbs_ini: Vec<i32> = (0..W[0].len())
            .map(|c| imbalance::max_imbalance(2, &part, W.par_iter().map(|w| w[c])))
            .collect();
        vn_first(&mut part, &W, 2);
        let imbs_end =
            (0..W[0].len()).map(|c| imbalance::max_imbalance(2, &part, W.par_iter().map(|w| w[c])));
        for (imb_ini, imb_end) in imbs_ini.into_iter().zip(imbs_end) {
            assert!(imb_end <= imb_ini);
            println!("imbalance : {} < {}", imb_end, imb_ini);
        }
    }

    const C: usize = 2;
    proptest!(
        /// vn_first should always improve balance !
        #[test]
        fn improve(
            (weights, mut partition) in
                (2..200_usize).prop_flat_map(|num_weights| {
                    let weights = prop::collection::vec(
                        (1..1000_i32, 1..1000_i32).prop_map(|(a, b)| [a, b]),
                        num_weights
                    );
                    let partition = prop::collection::vec(0..1_usize, num_weights);
                    (weights, partition)
                })
        ) {
            let imbs_ini: Vec<i32> = (0..C)
                .map(|c| imbalance::max_imbalance(2, &partition, weights.par_iter().map(|w| w[c])))
                .collect();
            vn_first(&mut partition, &weights, 2);
            let imbs_end: Vec<i32> = (0..C)
                .map(|c| imbalance::max_imbalance(2, &partition, weights.par_iter().map(|w| w[c])))
                .collect();
            for (imb_ini, imb_end) in imbs_ini.into_iter().zip(imbs_end) {
                proptest::prop_assert!(imb_end <= imb_ini);
                println!("imbalance : {} <= {}", imb_end, imb_ini);
            }
        }
    );
}
