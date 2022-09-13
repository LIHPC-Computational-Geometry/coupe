use crate::Error;
use itertools::Itertools as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::Sub;

fn vn_first<T>(partition: &mut [usize], weights: &[T], num_parts: usize) -> Result<usize, Error>
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
/// coupe::VnFirst
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
pub struct VnFirst;

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
        let part_count = 1 + *part_ids.par_iter().max().unwrap_or(&0);
        if part_count < 2 {
            return Ok(0);
        }
        vn_first(part_ids, weights, part_count)
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

        vn_first(&mut part, &W, 1).unwrap();
        let imb_ini = imbalance::imbalance(2, &part, W);
        vn_first(&mut part, &W, 2).unwrap();
        let imb_end = imbalance::imbalance(2, &part, W);
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
            vn_first(&mut partition, &weights, 2).unwrap();
            let imb_end = imbalance::max_imbalance(2, &partition, weights.par_iter().cloned());
            // Not sure if it is true for max_imbalance (i.e. weighter - lighter)
            proptest::prop_assert!(imb_end <= imb_ini);
        }
    );
}
