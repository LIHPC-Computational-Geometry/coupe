// use coupe::imbalance;
// use coupe::Topology;
use num_traits::Float;
use rand::distributions::Distribution;
// use rand::distributions::Uniform;
use rand::thread_rng;
use rand::SeedableRng as _;
use std::cmp;
use std::collections::HashMap;
use std::iter::Chain;
// use rand_distr::Distribution as _;
// use array_init;
use itertools::iproduct;
use itertools::Itertools;
use itertools::MinMaxResult::{MinMax, NoElements, OneElement};
use itertools::MultiProduct;
use itertools::Product;
use num_traits::ToPrimitive;
use num_traits::Unsigned;
use num_traits::Zero;
use num_traits::{FromPrimitive, Signed};
use rand::{distributions::Uniform, Rng};
use std::cmp::max;
use std::iter::Filter;
use std::iter::Map; // 0.6.5
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;

const NB_PARTS: usize = 2;
const NB_CRITERIA: usize = 3;
// const NB_WEIGHTS: usize = 100;
// const MIN_WEIGHT: usize = 1;
// const MIN_WEIGHTS: [usize; NB_CRITERIA] = [MIN_WEIGHT; NB_CRITERIA];
// const MAX_WEIGHT: usize = 1000;
// const MAX_WEIGHTS: [usize; NB_CRITERIA] = [MAX_WEIGHT; NB_CRITERIA];
// const NB_INTERVALS: usize = 10;

// TODO : See CKK

/// Trait alias for values accepted as weights by [PathOptimization].
pub trait Weight:
    Copy
    + std::fmt::Debug
    + Send
    + Sync
    + Sum
    + FromPrimitive
    + ToPrimitive
    + Zero
    + Sub<Output = Self>
    + PartialOrd
    // + AddAssign
    // + SubAssign
    + SignedNum
{
}

impl<T> Weight for T
where
    Self: Copy + std::fmt::Debug + Send + Sync,
    Self: Sum + PartialOrd + FromPrimitive + ToPrimitive + Zero,
    Self: Sub<Output = Self>,
    // + AddAssign + SubAssign,
    Self: SignedNum,
{
}

pub trait SignedNum: Sized {
    type SignedType: Copy + PartialOrd + Zero + AddAssign;

    fn to_signed(self) -> Self::SignedType;
}

macro_rules! impl_signednum_signed {
    ( $($t:ty),* ) => {
    $( impl SignedNum for $t
    {
        type SignedType = Self;

        fn to_signed(self) -> Self { self }
    }) *
    }
}

impl_signednum_signed! {i8, i16, i32, i64, i128, f32, f64}

macro_rules! impl_signednum_unsigned {
    ( $(($t:ty, $s:ty)),* ) => {
    $( impl SignedNum for $t
    {
        type SignedType = $s;

        fn to_signed(self) -> Self::SignedType { self.try_into().unwrap() }
    }) *
    }
}

impl_signednum_unsigned! {(u8,i8), (u16,i16), (u32,i32), (u64,i64), (u128,i128), (usize, isize)}

type CWeightId = usize;
type PartId = usize;
// type BoxIndices = [usize; NB_CRITERIA];
type BoxIndices = Vec<usize>;
pub struct Targetor {}

trait SearchStrat {
    fn new(nb_intervals: Vec<usize>) -> Self;
    // fn gen_indices(&self, origin: Vec<usize>, dist: usize) -> Box<dyn Iterator<Item = Vec<usize>>>;
    fn gen_indices(&self, origin: Vec<usize>, dist: usize) -> Box<dyn Iterator<Item = Vec<usize>>>;
    // fn gen_indices(&'a self, origin: Vec<usize>, dist: usize) -> Self::ItemIterator;
    // fn gen_indices(&self, origin: Vec<usize>, dist: usize) -> Box<dyn Iterator<Item = &isize>>;
    // found struct `std::iter::Map<std::iter::Filter<MultiProduct<std::ops::Range<isize>>, [closure@src/algorithms/targetor.rs:154:21: 154:35]>, [closure@src/algorithms/targetor.rs:155:18: 155:32]>`rustcClick for full compiler diagnostic
}

struct NeighborSearchStrat {
    nb_intervals: Vec<usize>,
}

impl SearchStrat for NeighborSearchStrat {
    fn new(nb_intervals: Vec<usize>) -> Self {
        let out = Self {
            nb_intervals: nb_intervals,
        };

        out
    }

    fn gen_indices(&self, origin: Vec<usize>, dist: usize) -> Box<dyn Iterator<Item = Vec<usize>>> {
        // fn gen_indices(&self, origin: Vec<usize>, dist: usize) -> Box<dyn Iterator<Item = &isize>> {
        //     self.coordinates
        //         .chunks_exact(self.dimension)
        //         .zip(self.node_refs.iter().cloned())
        // }
        // fn gen_indices(&self, origin: Vec<usize>, dist: usize) -> () {
        // where
        //     T: Iterator<Item = Vec<isize>>,

        // fn gen_indices<T>(&self, origin: Vec<usize>, dist: usize) -> T
        // where
        //     T: Iterator<Item = Vec<isize>>,
        // {
        let nb_criteria = origin.len();
        let mut left_bounds = vec![0; nb_criteria];
        let mut right_bounds = vec![0; nb_criteria];
        for criterion in 0..nb_criteria {
            left_bounds[criterion] = origin[criterion];
            right_bounds[criterion] = self.nb_intervals[criterion] - origin[criterion];
        }

        let mut rngs = Vec::new();
        for criterion in 0..nb_criteria as usize {
            let rng = -(cmp::min(dist, left_bounds[criterion]) as isize)
                ..=(cmp::min(dist, right_bounds[criterion]) as isize);
            rngs.push(rng);
        }

        let indices_generator = Box::new(
            rngs.into_iter()
                .multi_cartesian_product()
                .filter(move |indices| {
                    indices.iter().map(|i| i.abs() as usize).sum::<usize>() == dist
                })
                .map(move |indices| {
                    let mut box_indices = Vec::with_capacity(nb_criteria);
                    (0..nb_criteria).for_each(|criterion| {
                        box_indices.push((indices[criterion] + origin[criterion] as isize) as usize)
                    });
                    box_indices
                }),
        );

        // let mut out = Vec::with_capacity(nb_criteria);
        // (0..nb_criteria).for_each(|criterion| {
        //     out.push(
        //         (max_weights[criterion] - min_weights[criterion])
        //             .to_f64()
        //             .unwrap()
        //             / nb_intervals[criterion] as f64,
        //     );
        // });

        Box::new(indices_generator)
    }
}

struct RegularBoxHandler<W>
where
    W: Weight,
{
    // Values related to the instance to be improved
    min_weights: Vec<W>,
    max_weights: Vec<W>,
    // Parameters related to the discretization of the solution space
    nb_intervals: Vec<usize>,
    deltas: Vec<f64>,
    // Mapping used to search moves of specific gain from a discretization of the cweight space
    pub boxes: HashMap<BoxIndices, Vec<CWeightId>>,
}

impl<'a, W> RegularBoxHandler<W>
where
    W: Weight,
{
    fn new(cweights: &'a Vec<Vec<W>>, nb_intervals: Vec<usize>) -> Self {
        let mut boxes: HashMap<BoxIndices, Vec<CWeightId>> = HashMap::new();
        if cweights.len() <= 1 {
            panic!("expected more than one criteria weight");
        }

        let nb_criteria = cweights[0].len();
        let (mut min_weights, mut max_weights) =
            (vec![W::zero(); nb_criteria], vec![W::zero(); nb_criteria]);
        (0..nb_criteria).for_each(|criterion| {
            let (min, max) = cweights
                .iter()
                .minmax_by_key(|cweight| cweight[criterion])
                .into_option()
                .unwrap();
            min_weights[criterion] = min[criterion];
            max_weights[criterion] = max[criterion];
        });

        let deltas = Self::process_deltas(&min_weights, &max_weights, &nb_intervals);

        let mut out = Self {
            min_weights: min_weights,
            max_weights: max_weights,
            nb_intervals: nb_intervals,
            deltas: deltas,
            boxes: boxes,
        };

        cweights
            .iter()
            .enumerate()
            .for_each(|(cweight_id, cweight)| {
                let indices: BoxIndices = Self::init_box_indices(
                    cweight,
                    &out.min_weights,
                    &out.max_weights,
                    &out.nb_intervals,
                );
                match out.boxes.get_mut(&indices) {
                    Some(vect_box_indices) => vect_box_indices.push(cweight_id),
                    None => {
                        let vect_box_indices: Vec<usize> = vec![cweight_id];
                        out.boxes.insert(indices, vect_box_indices);
                    }
                }
            });

        out
    }

    fn process_deltas(
        min_weights: &Vec<W>,
        max_weights: &Vec<W>,
        nb_intervals: &Vec<usize>,
    ) -> Vec<f64> {
        let nb_criteria = min_weights.len();
        let mut out = Vec::with_capacity(nb_criteria);
        (0..nb_criteria).for_each(|criterion| {
            out.push(
                (max_weights[criterion] - min_weights[criterion])
                    .to_f64()
                    .unwrap()
                    / nb_intervals[criterion] as f64,
            );
        });

        out
    }

    fn init_box_indices(
        cweight: &Vec<W>,
        min_weights: &Vec<W>,
        max_weights: &Vec<W>,
        nb_intervals: &Vec<usize>,
    ) -> BoxIndices {
        let nb_criteria: usize = cweight.len();
        let mut deltas = Vec::with_capacity(nb_criteria);
        (0..nb_criteria).for_each(|criterion| {
            deltas.push(
                (max_weights[criterion] - min_weights[criterion])
                    .to_f64()
                    .unwrap()
                    / nb_intervals[criterion] as f64,
            );
        });

        let mut out: BoxIndices = vec![0; nb_criteria];
        cweight.iter().enumerate().for_each(|(criterion, weight)| {
            let diff: f64 = (*weight - min_weights[criterion]).to_f64().unwrap();
            let index: usize = cmp::min(
                (diff / deltas[criterion]).floor() as usize,
                nb_intervals[criterion] - 1,
            );
            out[criterion] = index;
        });
        out
    }

    fn box_indices(&self, cweight: Vec<W>) -> BoxIndices {
        let nb_criteria = cweight.len();

        let mut out: BoxIndices = vec![0; nb_criteria];
        cweight.iter().enumerate().for_each(|(criterion, weight)| {
            let diff: f64 = (*weight - self.min_weights[criterion]).to_f64().unwrap();
            let index: usize = cmp::min(
                (diff / self.deltas[criterion]).floor() as usize,
                self.nb_intervals[criterion] - 1,
            );
            out[criterion] = index;
        });

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Instance {
        pub cweights: Vec<Vec<usize>>,
        pub nb_intervals: Vec<usize>,
    }

    impl Instance {
        fn create_instance() -> Self {
            let mut out = Self {
                cweights: Vec::with_capacity(3),
                nb_intervals: Vec::with_capacity(2),
            };

            out.cweights.extend([vec![1, 1], vec![4, 2], vec![3, 5]]);
            out.nb_intervals.extend(vec![2, 2]);

            out
        }
    }

    #[test]
    fn check_regular_box_handler() {
        let instance = Instance::create_instance();

        let rbh = RegularBoxHandler::new(&instance.cweights, instance.nb_intervals);
        let mut expected_box_indices: Vec<BoxIndices> = Vec::with_capacity(3);
        expected_box_indices.extend([vec![0, 0], vec![1, 0], vec![1, 1]]);

        expected_box_indices
            .iter()
            .zip(instance.cweights)
            .for_each(|(box_indices, cweight)| {
                let indices = rbh.box_indices(cweight);
                assert!(
                    box_indices
                        .iter()
                        .zip(indices.iter())
                        .all(|(expected_val, computed_val)| expected_val == computed_val),
                    "Indices are not equal {:?}, {:?} ",
                    box_indices,
                    indices
                );
            });
    }

    #[test]
    fn check_neighbor_search_strat() {
        let instance = Instance::create_instance();

        let nss = NeighborSearchStrat::new(instance.nb_intervals);
        let origin = vec![1, 1];
        let dist = 1;
        let iterator_box_indices = nss.gen_indices(origin, dist);
        let mut expected_box_indices: Vec<BoxIndices> = Vec::with_capacity(3);
        expected_box_indices.extend([vec![0, 1], vec![1, 0], vec![2, 1], vec![1, 2]]);

        iterator_box_indices.for_each(|box_indices| {
            assert!(
                expected_box_indices
                    .iter()
                    .any(|iter_box_indices| box_indices
                        .iter()
                        .zip(iter_box_indices.iter())
                        .all(|(expected_val, computed_val)| expected_val == computed_val)),
                "Box indices {:?} was not found",
                box_indices,
            );
        })
    }
}

fn main() {}

pub struct PathOptimization {}
