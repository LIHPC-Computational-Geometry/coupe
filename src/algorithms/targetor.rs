use itertools::Itertools;
use num_traits::ToPrimitive;
use num_traits::Zero;
use num_traits::{FromPrimitive, Signed};
use std::cmp;
use std::collections::HashMap;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::Sub;

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
pub struct Targetor {}

struct BoxIndices<T>(Vec<T>);

struct IterBoxIndices<T> {
    inner: Box<dyn Iterator<Item = BoxIndices<T>>>,
}
impl<T> Iterator for IterBoxIndices<T> {
    type Item = BoxIndices<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
// impl<T> BoxIndices<T> {
//     fn iter(&self) -> IterBoxIndices<T> {
//         IterBoxIndices { inner: self }
//     }
// }

struct NeighborSearchStrat<T> {
    nb_intervals: Vec<T>,
}

// struct IterBoxIndices {
//     iter: Box<dyn Iterator<Item = BoxIndices>>,
// }

// impl Iterator for IterBoxIndices {
//     type Item = BoxIndices;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter.next()
//     }
// }

trait SearchStrat<U> {
    fn new(nb_intervals: Vec<U>) -> Self;
    fn gen_indices(&self, origin: &BoxIndices<U>, dist: U) -> IterBoxIndices<U>
    where
        U: Ord + Clone + Zero + std::ops::Sub<Output = U>;
}

impl<U> SearchStrat<U> for NeighborSearchStrat<U> {
    fn new(nb_intervals: Vec<U>) -> Self {
        let out = Self {
            nb_intervals: nb_intervals,
        };

        out
    }
    fn gen_indices(&self, origin: &BoxIndices<U>, dist: U) -> IterBoxIndices<U>
    where
        U: Ord + Clone + Zero + std::ops::Sub<Output = U>,
    {
        // fn gen_indices(&self, origin: BoxIndices, dist: usize) -> IterBoxIndices {
        let nb_criteria = origin.0.len();
        let mut left_bounds = vec![U::zero(); nb_criteria];
        let mut right_bounds = vec![U::zero(); nb_criteria];
        for criterion in 0..nb_criteria {
            left_bounds[criterion] = origin.0[criterion];
            right_bounds[criterion] = self.nb_intervals[criterion] - origin.0[criterion];
        }

        let mut rngs = Vec::new();
        for criterion in 0..nb_criteria as usize {
            let rng = -(cmp::min(dist, left_bounds[criterion]) as isize)
                ..=(cmp::min(dist, right_bounds[criterion]) as isize);
            rngs.push(rng);
        }

        let indices_generator = rngs
            .into_iter()
            .multi_cartesian_product()
            .filter(move |indices| indices.iter().map(|i| i.abs() as usize).sum::<usize>() == dist)
            .map(move |indices| {
                let mut box_indices = Vec::with_capacity(nb_criteria);
                (0..nb_criteria).for_each(|criterion| {
                    box_indices.push((indices[criterion] + origin[criterion] as isize) as usize)
                });
                BoxIndices::from(box_indices)
            });

        IterBoxIndices {
            inner: Box::new(indices_generator.into_iter()),
        }
    }
}

// struct RegularBoxHandler<W, T>
// where
//     W: Weight,
// {
//     // Values related to the instance to be improved
//     min_weights: Vec<W>,
//     max_weights: Vec<W>,
//     // Parameters related to the discretization of the solution space
//     nb_intervals: Vec<usize>,
//     deltas: Vec<f64>,
//     // Mapping used to search moves of specific gain from a discretization of the cweight space
//     pub boxes: HashMap<BoxIndices<T>, Vec<CWeightId>>,
// }

// impl<'a, W, T> RegularBoxHandler<W, T>
// where
//     W: Weight,
// {
//     fn new(cweights: &'a Vec<Vec<W>>, nb_intervals: Vec<usize>) -> Self {
//         let mut boxes: HashMap<BoxIndices, Vec<CWeightId>> = HashMap::new();
//         if cweights.len() <= 1 {
//             panic!("expected more than one criteria weight");
//         }

//         let nb_criteria = cweights[0].len();
//         let (mut min_weights, mut max_weights) =
//             (vec![W::zero(); nb_criteria], vec![W::zero(); nb_criteria]);
//         (0..nb_criteria).for_each(|criterion| {
//             let (min, max) = cweights
//                 .iter()
//                 .minmax_by_key(|cweight| cweight[criterion])
//                 .into_option()
//                 .unwrap();
//             min_weights[criterion] = min[criterion];
//             max_weights[criterion] = max[criterion];
//         });

//         let deltas = Self::process_deltas(&min_weights, &max_weights, &nb_intervals);

//         let mut out = Self {
//             min_weights: min_weights,
//             max_weights: max_weights,
//             nb_intervals: nb_intervals,
//             deltas: deltas,
//             boxes: boxes,
//         };

//         cweights
//             .iter()
//             .enumerate()
//             .for_each(|(cweight_id, cweight)| {
//                 let indices: BoxIndices = Self::init_box_indices(
//                     cweight,
//                     &out.min_weights,
//                     &out.max_weights,
//                     &out.nb_intervals,
//                 );
//                 match out.boxes.get_mut(&indices) {
//                     Some(vect_box_indices) => vect_box_indices.push(cweight_id),
//                     None => {
//                         let vect_box_indices: Vec<usize> = vec![cweight_id];
//                         out.boxes.insert(indices, vect_box_indices);
//                     }
//                 }
//             });

//         out
//     }

//     fn process_deltas(
//         min_weights: &Vec<W>,
//         max_weights: &Vec<W>,
//         nb_intervals: &Vec<usize>,
//     ) -> Vec<f64> {
//         let nb_criteria = min_weights.len();
//         let mut out = Vec::with_capacity(nb_criteria);
//         (0..nb_criteria).for_each(|criterion| {
//             out.push(
//                 (max_weights[criterion] - min_weights[criterion])
//                     .to_f64()
//                     .unwrap()
//                     / nb_intervals[criterion] as f64,
//             );
//         });

//         out
//     }

//     fn init_box_indices<T>(
//         cweight: &Vec<W>,
//         min_weights: &Vec<W>,
//         max_weights: &Vec<W>,
//         nb_intervals: &Vec<usize>,
//     ) -> BoxIndices<T> {
//         let nb_criteria: usize = cweight.len();
//         let mut deltas = Vec::with_capacity(nb_criteria);
//         (0..nb_criteria).for_each(|criterion| {
//             deltas.push(
//                 (max_weights[criterion] - min_weights[criterion])
//                     .to_f64()
//                     .unwrap()
//                     / nb_intervals[criterion] as f64,
//             );
//         });

//         let mut out: BoxIndices = vec![0; nb_criteria];
//         cweight.iter().enumerate().for_each(|(criterion, weight)| {
//             let diff: f64 = (*weight - min_weights[criterion]).to_f64().unwrap();
//             let index: usize = cmp::min(
//                 (diff / deltas[criterion]).floor() as usize,
//                 nb_intervals[criterion] - 1,
//             );
//             out[criterion] = index;
//         });
//         out
//     }

//     fn box_indices<T>(&self, cweight: Vec<W>) -> BoxIndices<T> {
//         let nb_criteria = cweight.len();

//         let mut out: BoxIndices = vec![0; nb_criteria];
//         cweight.iter().enumerate().for_each(|(criterion, weight)| {
//             let diff: f64 = (*weight - self.min_weights[criterion]).to_f64().unwrap();
//             let index: usize = cmp::min(
//                 (diff / self.deltas[criterion]).floor() as usize,
//                 self.nb_intervals[criterion] - 1,
//             );
//             out[criterion] = index;
//         });

//         out
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     struct Instance {
//         pub cweights: Vec<Vec<usize>>,
//         pub nb_intervals: Vec<usize>,
//     }

//     impl Instance {
//         fn create_instance() -> Self {
//             let mut out = Self {
//                 cweights: Vec::with_capacity(3),
//                 nb_intervals: Vec::with_capacity(2),
//             };

//             out.cweights.extend([vec![1, 1], vec![4, 2], vec![3, 5]]);
//             out.nb_intervals.extend(vec![2, 2]);

//             out
//         }
//     }

//     #[test]
//     fn check_regular_box_handler() {
//         let instance = Instance::create_instance();

//         let rbh = RegularBoxHandler::new(&instance.cweights, instance.nb_intervals);
//         let mut expected_box_indices: Vec<BoxIndices> = Vec::with_capacity(3);
//         expected_box_indices.extend([vec![0, 0], vec![1, 0], vec![1, 1]]);

//         expected_box_indices
//             .iter()
//             .zip(instance.cweights)
//             .for_each(|(box_indices, cweight)| {
//                 let indices = rbh.box_indices(cweight);
//                 assert!(
//                     box_indices
//                         .iter()
//                         .zip(indices.iter())
//                         .all(|(expected_val, computed_val)| expected_val == computed_val),
//                     "Indices are not equal {:?}, {:?} ",
//                     box_indices,
//                     indices
//                 );
//             });
//     }

//     #[test]
//     fn check_neighbor_search_strat() {
//         let instance = Instance::create_instance();

//         let nss = NeighborSearchStrat::new(instance.nb_intervals);
//         let origin = vec![1, 1];
//         let dist = 1;
//         let iterator_box_indices = nss.gen_indices(origin, dist);
//         let mut expected_box_indices: Vec<BoxIndices> = Vec::with_capacity(3);
//         expected_box_indices.extend([vec![0, 1], vec![1, 0], vec![2, 1], vec![1, 2]]);

//         iterator_box_indices.for_each(|box_indices| {
//             assert!(
//                 expected_box_indices
//                     .iter()
//                     .any(|iter_box_indices| box_indices
//                         .iter()
//                         .zip(iter_box_indices.iter())
//                         .all(|(expected_val, computed_val)| expected_val == computed_val)),
//                 "Box indices {:?} was not found",
//                 box_indices,
//             );
//         })
//     }
// }

fn main() {}
