use core::num::fmt::Part;
use itertools::Itertools;
use num_traits::ToPrimitive;
use num_traits::{FromPrimitive, PrimInt, Signed, Zero};
use std::alloc::System;
use std::cmp;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::ops::{Add, Div, Mul, Neg, Sub, SubAssign};
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
    // + PrimInt
{
}

impl<T> Weight for T
where
    Self: Copy + std::fmt::Debug + Send + Sync,
    Self: Sum + PartialOrd + FromPrimitive + ToPrimitive + Zero,
    // Self: PrimInt,
    Self: Sub<Output = Self>,
    // + AddAssign + SubAssign,
    Self: SignedNum,
{
}

pub trait SignedNum: Sized {
    type SignedType: Copy + PartialOrd + Zero + AddAssign + Debug;

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
type CriterionId = usize;
type CWeightMove = (CWeightId, PartId);
// pub struct Targetor {}

// struct BoxIndices<T>(Vec<T>);

// struct IterBoxIndices<T> {
//     inner: Box<dyn Iterator<Item = BoxIndices<T>>>,
// }
// impl<T> Iterator for IterBoxIndices<T> {
//     type Item = BoxIndices<T>;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.inner.next()
//     }
// }

// // impl<T> BoxIndices<T> {
// //     fn iter(&self) -> IterBoxIndices<T> {
// //         IterBoxIndices { inner: self }
// //     }
// // }

// struct NeighborSearchStrat<T> {
//     nb_intervals: Vec<T>,
// }

// struct IterBoxIndices {
//     iter: Box<dyn Iterator<Item = BoxIndices>>,
// }

// impl Iterator for IterBoxIndices {
//     type Item = BoxIndices;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter.next()
//     }
// }

// pub trait BoxIndex:
//     Clone + Copy + PartialEq + PartialOrd + Add<Self, Output = Self> + Sub<Self, Output = Option<Self>>
// {
//     fn new(value: T) -> Option<Self>;
//     fn value(self) -> T;
// }

/// Trait alias for values accepted as box indices by [...].
// pub trait BoxIndex:
//     Copy
//     + std::fmt::Debug
//     + Send
//     + Sync
//     + Sum
//     + PartialOrd
//     + FromPrimitive
//     + ToPrimitive
//     + Zero
//     + Sub<Output = Self>
//     + AddAssign
//     + SignedNum
//     + Into<isize>
// {
// }
use num_traits::Num;

use crate::Error;

#[derive(Debug)]
struct StrictlyPositiveError;

impl fmt::Display for StrictlyPositiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StrictlyPositive only accepts values greater than zero!")
    }
}
impl std::error::Error for StrictlyPositiveError {}

// Trait for strictly positive values
pub trait StrictlyPositive: Num + PartialOrd<Self> + Zero {
    fn is_strictly_positive(&self) -> bool {
        *self > Self::zero()
    }

    fn from_value(value: Self) -> Result<Self, StrictlyPositiveError> {
        if value > Self::zero() {
            Ok(value)
        } else {
            Err(StrictlyPositiveError)
        }
    }

    fn to_value(&self) -> Self {
        *self
    }
}

impl<T: Num + PartialOrd<T> + Zero> StrictlyPositive for T {}

// // Trait alias for values accepted as indices for the solution space discretization.
pub trait PositiveInteger: PrimInt + Zero + Add<Self, Output = Self> {}

impl<T: PrimInt + Zero + Add<Output = Self>> PositiveInteger for T {}

// // Structure encapsulating the indices associated with a box in the discretised solution space
#[derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct BoxIndices<T>
where
    T: PositiveInteger,
{
    indices: Vec<T>,
}
impl<T> BoxIndices<T>
where
    T: PositiveInteger,
{
    fn new(indices: Vec<T>) -> Self {
        Self { indices }
    }
}

struct IterBoxIndices<'a, T>
where
    T: PositiveInteger,
{
    inner: Box<dyn Iterator<Item = BoxIndices<T>> + 'a>,
}
impl<'a, T> Iterator for IterBoxIndices<'a, T>
where
    T: PositiveInteger,
{
    type Item = BoxIndices<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

trait SearchStrat<'a, T: PositiveInteger> {
    fn new(nb_intervals: Vec<T>) -> Self;
    fn gen_indices(&'a self, origin: &'a BoxIndices<T>, dist: T) -> IterBoxIndices<'a, T>;
}

struct NeighborSearchStrat<T>
where
    T: PositiveInteger,
{
    nb_intervals: Vec<T>,
}

impl<'a, T: PositiveInteger> SearchStrat<'a, T> for NeighborSearchStrat<T>
where
    isize: From<T>,
{
    fn new(nb_intervals: Vec<T>) -> Self {
        let out = Self {
            nb_intervals: nb_intervals,
        };

        out
    }

    fn gen_indices(&self, origin: &'a BoxIndices<T>, dist: T) -> IterBoxIndices<'a, T> {
        let nb_criteria = origin.indices.len();
        let mut left_bounds = vec![T::zero(); nb_criteria];
        let mut right_bounds = vec![T::zero(); nb_criteria];
        for criterion in 0..nb_criteria {
            left_bounds[criterion] = origin.indices[criterion];
            right_bounds[criterion] = self.nb_intervals[criterion] - origin.indices[criterion];
        }

        let mut rngs = Vec::new();
        for criterion in 0..nb_criteria as usize {
            let rng = -(isize::from(cmp::min(dist, left_bounds[criterion])))
                ..=isize::from(cmp::min(dist, right_bounds[criterion]));
            rngs.push(rng);
        }

        let indices_generator = rngs
            .into_iter()
            .multi_cartesian_product()
            .filter(move |indices| {
                indices.iter().map(|i| i.abs()).sum::<isize>() == isize::from(dist)
            })
            .map(move |indices| {
                let mut box_indices = Vec::with_capacity(nb_criteria);
                (0..nb_criteria).for_each(|criterion| {
                    box_indices
                        .push(T::from(indices[criterion]).unwrap() + origin.indices[criterion])
                });
                BoxIndices::new(box_indices)
            });

        IterBoxIndices {
            inner: Box::new(indices_generator.into_iter()),
        }
    }
}

// // Trait alias for handling boxes of the solution space discretization.
// trait BoxHandler<T: PositiveInteger, W: Weight> {
//     fn new(cweights: &Vec<Vec<W>>, nb_intervals: Vec<T>) -> Box<Self>;
//     // where
//     //     Self: Sized;
//     fn box_indices(&self, cweight: &Vec<W>) -> BoxIndices<T>;
// }
trait BoxHandler<'a, T: PositiveInteger, W: Weight> {
    // fn new(cweights: &Vec<Vec<W>>, nb_intervals: Vec<T>) -> Box<dyn BoxHandler<T, W>>;
    // fn new(cweights: &Vec<Vec<W>>, nb_intervals: Vec<T>) -> Box<Self>;
    fn box_indices(&self, cweight: &'a Vec<W>) -> BoxIndices<T>;
    fn filter_cweights(
        &self,
        box_indices: &'a BoxIndices<T>,
        part_source: PartId,
        partition_imbalance: W,
        partition: &'a Vec<PartId>,
        cweights: &Vec<Vec<W>>,
    ) -> IterBoxIndices<'a, T>;
}

// struct BoxHandlerWrapper<W: Weight, T: PositiveInteger, H: BoxHandler<W, T>>(
//     Box<H>,
//     PhantomData<T>,
//     PhantomData<W>,
// );

// impl<W: Weight, T: PositiveInteger, H: BoxHandler<W, T>> BoxHandler<W, T>
//     for BoxHandlerWrapper<W, T, H>
// {
//     fn new(cweights: &Vec<Vec<W>>, nb_intervals: Vec<T>) -> Box<Self>
//     where
//         Self: Sized,
//     {
//         Box::new(BoxHandlerWrapper(H::new(cweights, nb_intervals), PhantomData, PhantomData))
//     }

//     fn box_indices(&self, cweight: &Vec<W>) -> BoxIndices<T> {
//         self.0.box_indices(cweight)
//     }
// }
// // Trait alias for handling regular delta of the solution space discretization.
trait RegularDeltaHandler<'a, T: PositiveInteger, W: Weight> {
    fn process_deltas(
        min_weights: &Vec<W>,
        max_weights: &Vec<W>,
        nb_intervals: &Vec<T>,
    ) -> Vec<f64>;
}

impl<'a, T: PositiveInteger, W: Weight> RegularDeltaHandler<'a, T, W> for RegularBoxHandler<T, W>
where
    isize: From<T>,
    f64: From<T>,
{
    fn process_deltas(
        min_weights: &Vec<W>,
        max_weights: &Vec<W>,
        nb_intervals: &Vec<T>,
    ) -> Vec<f64> {
        let nb_criteria = min_weights.len();
        let mut out = Vec::with_capacity(nb_criteria);
        (0..nb_criteria).for_each(|criterion| {
            out.push(
                (max_weights[criterion] - min_weights[criterion])
                    .to_f64()
                    .unwrap()
                    / f64::from(nb_intervals[criterion]),
            );
        });

        out
    }
}

struct RegularBoxHandler<T, W>
where
    T: PositiveInteger,
    W: Weight,
{
    // Values related to the instance to be improved
    min_weights: Vec<W>,
    // max_weights: Vec<W>,
    // Parameters related to the discretization of the solution space
    nb_intervals: Vec<T>,
    deltas: Vec<f64>,
    // Mapping used to search moves of specific gain from a discretization of the cweight space
    pub boxes: BTreeMap<BoxIndices<T>, Vec<CWeightId>>,
}

impl<T, W> RegularBoxHandler<T, W>
where
    T: PositiveInteger,
    W: Weight,
    f64: From<T>,
    isize: From<T>,
{
    fn new(cweights: &Vec<Vec<W>>, nb_intervals: Vec<T>) -> Self {
        let boxes: BTreeMap<BoxIndices<T>, Vec<CWeightId>> = BTreeMap::new();
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
            // max_weights: max_weights,
            nb_intervals: nb_intervals,
            deltas: deltas,
            boxes: boxes,
        };

        cweights
            .iter()
            .enumerate()
            .for_each(|(cweight_id, cweight)| {
                let indices: BoxIndices<T> = Self::box_indices(&out, cweight);
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
}

// impl<'a, W: Weight, T: PositiveInteger> RegularBoxHandler<W, T>
impl<'a, T: PositiveInteger, W: Weight> BoxHandler<'a, T, W> for RegularBoxHandler<T, W>
where
    isize: From<T>,
    f64: From<T>,
{
    fn box_indices(&self, cweight: &Vec<W>) -> BoxIndices<T> {
        let nb_criteria: usize = cweight.len();

        let mut out: BoxIndices<T> = BoxIndices::new(vec![T::zero(); nb_criteria]);
        cweight.iter().enumerate().for_each(|(criterion, weight)| {
            let diff: f64 = (*weight - self.min_weights[criterion]).to_f64().unwrap();
            let index = if diff >= 0. {
                cmp::min(
                    (diff / self.deltas[criterion]).floor() as isize,
                    isize::from(self.nb_intervals[criterion]) - 1,
                )
            } else {
                0
            };
            //     //     imbalances[criterion][part_source]
            //     //         .try_into()
            //     //         .unwrap_or(W::zero())
            //     // } else {
            //     //     W::zero()
            //     // };
            // if diff < 0 {

            // } else {
            //     let index: isize = cmp::min(
            //         (diff / self.deltas[criterion]).floor() as isize,
            //         isize::from(self.nb_intervals[criterion]) - 1,
            //     );
            // }
            // println!("diff : {}, {}", diff, index);
            out.indices[criterion] = T::from(index).unwrap();
        });
        out
    }

    fn filter_cweights(
        &self,
        box_indices: &'a BoxIndices<T>,
        part_source: PartId,
        partition_imbalance: W,
        partition: &'a Vec<PartId>,
        cweights: &Vec<Vec<W>>,
    ) -> IterBoxIndices<'a, T> {
        let mut candidates = self.boxes.get(&box_indices);

        candidates.filter(predicate)
    }
    //

    //
    //
}

trait Repartitioning<'a, T, W> {
    // TODO: Add a builder for generic Repartitioning
    fn find_move(
        &self,
        // partition: &'a Vec<PartId>,
        cweights: &'a Vec<Vec<W>>,
        min_weights: &'a Vec<W>,
        max_weights: &'a Vec<W>,
        // parts_target_loads: &'a Vec<Vec<W>>,
    ) -> CWeightMove;
}

trait PartitionImbalanceHandler<'a, T, W>
where
    W: Weight,
    <W as SignedNum>::SignedType: Signed + TryFrom<W> + Copy,
{
    // fn compute_partition_imbalances(
    //     partition: &'a Vec<PartId>,
    //     cweights: &'a Vec<Vec<W>>,
    //     parts_target_loads: &'a Vec<Vec<W>>,
    // ) -> Vec<Vec<isize>>;
    // fn compute_partition_imbalances(
    //     partition: &'a Vec<PartId>,
    //     cweights: &'a Vec<Vec<W>>,
    //     parts_target_loads: &'a Vec<Vec<W>>,
    // ) -> Vec<Vec<W::SignedType>>;

    fn compute_partition_imbalances(&self, cweights: &'a Vec<Vec<W>>) -> Vec<Vec<W::SignedType>>;

    fn process_imbalance(
        &self,
        // partition: &'a Vec<PartId>,
        cweights: &'a Vec<Vec<W>>,
        // parts_target_loads: &'a Vec<Vec<W>>,
    ) -> (CriterionId, PartId);
}

#[derive(Debug)]
pub struct Targetor<'a, T, W>
where
    T: PositiveInteger,
    W: Weight,
    // for<'a> B: BoxHandler<W, T>,
{
    // Instance data
    nb_parts: usize,
    nb_intervals: &'a Vec<T>,
    deltas: &'a Vec<W>,
    parts_target_loads: &'a Vec<Vec<W>>,

    // Partition state
    partition: &'a Vec<PartId>,
    // parts_load_per_crit: Vec<W>,
    // parts_imbalances_per_crit: &'a Vec<Vec<isize>>,
    // Algorithm inner data struct
    // box_handler: dyn BoxHandler<'a, W, T>,
    // boxes_mapping: BTreeMap<BoxIndices<T>, Vec<CWeightId>>,
    // box_handler: Box<dyn BoxHandler<T, W>>,
    box_handler: Box<dyn BoxHandler<'a, T, W>>,
}

impl<'a, T, W> Debug for dyn BoxHandler<'a, T, W>
where
    T: PositiveInteger,
    W: Weight,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "BoxHandler")
    }
}
// struct IterBoxIndices<'a, T>
// where
//     T: PositiveInteger,
// {
//     inner: Box<dyn Iterator<Item = BoxIndices<T>> + 'a>,
// }

// impl fmt for Targetor<'a, T, W> {
// }

impl<'a, T: PositiveInteger, W: Weight> PartitionImbalanceHandler<'a, T, W> for Targetor<'a, T, W>
where
    // usize: From<T>,
    // isize: From<W>,
    // usize: From<T>,
    // isize: Sub<W>,
    // Vec<Vec<isize>>: From<Vec<Vec<W>>>,
    <W as SignedNum>::SignedType: Signed + TryFrom<W> + Copy,
    // T: 'static,
    // W: 'static,
{
    fn compute_partition_imbalances(
        &self,
        // partition: &'a Vec<PartId>,
        cweights: &'a Vec<Vec<W>>,
        // parts_target_loads: &'a Vec<Vec<W>>,
    ) -> Vec<Vec<W::SignedType>> {
        // Compute part loads

        let nb_criterion = self.parts_target_loads.len();
        let mut init = vec![vec![W::SignedType::zero(); 2]; nb_criterion];
        for criterion in 0..nb_criterion {
            for part in 0..2 {
                init[criterion][part] = -self.parts_target_loads[criterion][part].to_signed();
            }
        }

        let res: Vec<Vec<W::SignedType>> =
            self.partition
                .iter()
                .zip(cweights)
                .fold(init, |mut acc, (&part, w)| {
                    acc[part] = acc[part]
                        .iter()
                        .zip(w)
                        .map(|(acc_val, w_val)| *acc_val + (*w_val).to_signed())
                        .collect();
                    acc
                });
        res
    }

    /* Function returning a couple composed of the most imbalanced criterion
    and the part from which a weight should me moved.
     */
    fn process_imbalance(
        &self,
        // partition: &'a Vec<PartId>,
        cweights: &'a Vec<Vec<W>>,
        // parts_target_loads: &'a Vec<Vec<W>>,
    ) -> (CriterionId, PartId) {
        let parts_imbalances_per_crit = self.compute_partition_imbalances(cweights);

        let mut max_imb = W::SignedType::zero();

        let mut res: (CriterionId, PartId) = (CriterionId::zero(), PartId::zero());
        parts_imbalances_per_crit
            .iter()
            .enumerate()
            .for_each(|(criterion, loads)| {
                let max = loads[0].abs();
                if max > max_imb {
                    let part = 1 - (loads[0] >= loads[1]) as usize;
                    res = (criterion, part);
                    max_imb = max;
                }
            });

        res
    }
}
// fn new(
//     partition: &'a Vec<PartId>,
//     cweights: &'a Vec<Vec<W>>,
//     // Targetor specific parameter
//     nb_intervals: &'a Vec<T>,
//     deltas: &'a Vec<W>,
//     parts_target_loads: &'a Vec<Vec<W>>,

//     // parts_load_per_crit: Vec<W>,
//     parts_imbalances_per_crit: &'a Vec<Vec<isize>>,
// ) -> Self;

impl<'a, T: PositiveInteger, W: Weight> Repartitioning<'a, T, W> for Targetor<'a, T, W>
where
    isize: From<T>,
    //     isize: From<W>,
    //     f64: From<T>,
    //     Vec<Vec<isize>>: From<Vec<Vec<W>>>,
    // usize: From<T>,
    <W as SignedNum>::SignedType: Signed + TryFrom<W> + Copy + TryInto<W>,
    // T: 'static,
    // W: 'static,
{
    fn find_move(
        &self,
        // partition: &'a Vec<PartId>,
        cweights: &'a Vec<Vec<W>>,
        min_weights: &'a Vec<W>,
        max_weights: &'a Vec<W>,
        // parts_target_loads: &'a Vec<Vec<W>>,
    ) -> CWeightMove {
        // Setup target part and most imbalanced criteria
        let (most_imb_criterion, part_source) = self.process_imbalance(cweights);

        // Setup search strat
        let search_strat = NeighborSearchStrat::new(self.nb_intervals.clone());

        // Setup target gain
        let nb_criteria = self.parts_target_loads.len();
        let criterion_rng = 0..nb_criteria;
        let imbalances = self.compute_partition_imbalances(cweights);

        let mut target_gain = vec![W::zero(); nb_criteria];
        criterion_rng.for_each(|criterion| {
            let val = imbalances[criterion][part_source];
            if val > <W as SignedNum>::SignedType::zero() {
                target_gain[criterion] = val.try_into().unwrap_or(W::zero());
            }
            // let criterion_target_gain = if val > <W as SignedNum>::SignedType::zero() {
            //     imbalances[criterion][part_source]
            //         .try_into()
            //         .unwrap_or(W::zero())
            // } else {
            //     W::zero()
            // };
            // target_gain.push(criterion_target_gain);
            // let val = imbalances[criterion][part_source];
            // target_gain[criterion] = if val > <W as SignedNum>::SignedType::zero() {
            //     imbalances[criterion][part_source].try_into()
            // } else {
            //     W::zero()
            // };

            // let val = imbalances[criterion][part_source]
            //     .try_into()
            //     .unwrap_or(W::zero());
            // // .unwrap();
            // println!("val {:?}", val);
            // target_gain.push(val);
            // // let mut val = W::SignedType::zero();
            // if imbalances[criterion][part_source] > val {
            //     val = imbalances[criterion][part_source];
            // }

            // target_gain[criterion] = val;
            // // let val = imbalances[criterion][part_source].abs();
            // // if val <= max_weights[criterion].to_signed() {
            // //     target_gain[criterion] = val;
            // // } else {
            // //     target_gain[criterion] = max_weights[criterion].to_signed();
            // // }
        });

        // Retrieve candidates
        // let cweight = target_gain.try_into().unwrap();
        let target_box_indices = self.box_handler.box_indices(&target_gain);

        (0, 0)
    }
}

impl<'a, T: PositiveInteger, W: Weight> Targetor<'a, T, W>
where
    isize: From<T>,
    // isize: From<W>,
    f64: From<T>,
    // Vec<Vec<isize>>: From<Vec<Vec<W>>>,
    // <W as targetor_bipart_multicrit::SignedNum>::SignedType: Signed`
    // <W as targetor_bipart_multicrit::SignedNum>::SignedType: TryFrom<W>`
    // `usize: From<T>`
    <W as SignedNum>::SignedType: Signed + TryFrom<W> + Copy + TryInto<W>,
    // usize: From<T>,
    T: 'static,
    W: 'static,
{
    fn new(
        partition: &'a Vec<PartId>,
        cweights: &'a Vec<Vec<W>>,
        // Targetor specific parameter
        nb_intervals: &'a Vec<T>,
        deltas: &'a Vec<W>,
        parts_target_loads: &'a Vec<Vec<W>>,
        // box_handler: &'a dyn BoxHandler<T,,
        // parts_load_per_crit: Vec<W>,
        // parts_imbalances_per_crit: &'a Vec<Vec<isize>>,
    ) -> Self {
        // let parts_imbalances_per_crit = Self::process_partition_imbalances(&partition, &cweights, &parts_target_loads);
        // let box_handler = RegularBoxHandler::new(cweights, vec![T::zero()]);
        let nb_criteria = parts_target_loads.len();

        let rbh = RegularBoxHandler::new(cweights, nb_intervals.to_vec());

        let mut out = Self {
            // Instance data
            nb_parts: 0,
            nb_intervals: nb_intervals,
            parts_target_loads: parts_target_loads,
            deltas: deltas,

            // Partition state
            partition: partition,
            box_handler: Box::new(rbh),
            // parts_load_per_crit: Vec<W>,
            // Vec<Vec<W::SignedType>>
            // parts_imbalances_per_crit: &Vec::with_capacity(nb_criteria),
            // Algorithm inner data struct
            // box_handler: RegularBoxHandler::new(&cweights, nb_intervals),
            // box_handler: RegularBoxHandler::new(cweights, vec![T::zero()])
            // box_handler: RegularBoxHandler::new(cweights, vec![T::zero()]),
            // box_handler: Box::new(RegularBoxHandler {
            //     min_weights: vec![W::zero()],
            //     // Parameters related to the discretization of the solution space
            //     nb_intervals: vec![T::zero()],
            //     deltas: vec![0.],
            //     // Mapping used to search moves of specific gain from a discretization of the cweight space
            //     boxes: BTreeMap::new(),
            // }),
        };

        // let parts_imbalances_per_crit = out.compute_partition_imbalances(cweights);
        // out.parts_imbalances_per_crit = &parts_imbalances_per_crit;

        out
    }

    // fn find_move(partition: &'a Vec<PartId>) -> CWeightMove;
}
// partition
//         .par_iter()
//         .zip(weights)
//         .fold(
//             || vec![W::Item::zero(); num_parts],
//             |mut acc, (&part, w)| {
//                 acc[part] += w;
//                 acc
//             },
//         )
//         .reduce_with(|mut weights0, weights1| {
//             for (w0, w1) in weights0.iter_mut().zip(weights1) {
//                 *w0 += w1;
//             }
//             weights0
//         })
//         .unwrap_or_else(|| vec![W::Item::zero(); num_parts])

#[cfg(test)]
mod tests {

    use super::*;

    struct Instance {
        pub cweights: Vec<Vec<f64>>,
        pub nb_intervals: Vec<i16>,
    }

    impl Instance {
        fn create_instance() -> Self {
            // let mut out = Self {
            //     cweights: Vec::with_capacity(3),
            //     nb_intervals: Vec::with_capacity(2),
            // };

            // out.cweights = vec![vec![1.5, 0.5], vec![2.5, 2.5], vec![3.5, 1.5]];
            // out.nb_intervals.extend(vec![2, 2]);
            // out.cweights = vec![vec![1.5, 0.5], vec![2.5, 2.5], vec![3.5, 1.5]];
            // out.nb_intervals.extend(vec![2, 2]);
            let out = Self {
                cweights: vec![vec![0., 2.], vec![1.5, 0.5], vec![4., 1.5]],
                nb_intervals: vec![4, 3],
            };

            out
        }
    }

    // #[test]
    // fn check_regular_box_handler() {
    //     let instance = Instance::create_instance();

    //     // Split with steps 1.0 on the first criterion and 0.5 on the second one.
    //     let rbh = RegularBoxHandler::new(&instance.cweights, instance.nb_intervals);
    //     let mut expected_box_indices: Vec<Vec<i16>> = Vec::with_capacity(3);
    //     expected_box_indices.extend([vec![0, 2], vec![1, 0], vec![3, 2]]);

    //     expected_box_indices
    //         .iter()
    //         .zip(instance.cweights)
    //         .for_each(|(box_indices, cweight)| {
    //             let values = rbh.box_indices(&cweight);
    //             assert!(
    //                 box_indices
    //                     .iter()
    //                     .zip(values.indices.iter())
    //                     .all(|(expected_val, computed_val)| expected_val == computed_val),
    //                 "Indices are not equal {:?}, {:?} ",
    //                 box_indices,
    //                 values
    //             );
    //         });
    // }

    // #[test]
    // fn check_neighbor_search_strat() {
    //     let nss: NeighborSearchStrat<i16> = NeighborSearchStrat::new(vec![2, 2]);
    //     let origin = BoxIndices::new(vec![1, 1]);
    //     let dist = 1;
    //     let mut expected_box_indices = Vec::with_capacity(3);
    //     expected_box_indices.extend([vec![0, 1], vec![1, 0], vec![2, 1], vec![1, 2]]);

    //     let iterator_box_indices = nss.gen_indices(&origin, dist);

    //     iterator_box_indices.for_each(|box_indices| {
    //         assert!(
    //             expected_box_indices
    //                 .iter()
    //                 .any(|iter_box_indices| box_indices
    //                     .indices
    //                     .iter()
    //                     .zip(iter_box_indices.iter())
    //                     .all(|(expected_val, computed_val)| expected_val == computed_val)),
    //             "Box indices {:?} was not found",
    //             box_indices,
    //         );
    //     })
    // }

    #[test]
    fn check_targetor() {
        let instance = Instance::create_instance();
        let mut partition = Vec::with_capacity(3);
        partition.extend([0, 0, 0]);
        let box_handler = RegularBoxHandler::new(&instance.cweights, instance.nb_intervals);
        let parts_target_loads: Vec<Vec<f64>> = vec![vec![3., 4.], vec![2.5, 0.]];

        // cweights: vec![vec![0., 2.], vec![1.5, 0.5], vec![4., 1.5]],
        let min_weights = vec![0., 0.5];
        let max_weights = vec![4., 2.];
        let deltas = RegularBoxHandler::process_deltas(
            &min_weights,
            &max_weights,
            &box_handler.nb_intervals,
        );
        // box_handler::process_deltas(&, &max_weights, &nb_intervals);
        let targetor = Targetor::new(
            &partition,
            &instance.cweights,
            &box_handler.nb_intervals,
            &deltas,
            &parts_target_loads,
            // parts_imbalances_per_crit
        );
        // cweights: &'a Vec<Vec<W>>,
        // min_weights: &'a Vec<W>,
        // max_weights: &'a Vec<W>,
        let mv = targetor.find_move(&instance.cweights, &min_weights, &max_weights);
        // println!("Move: {:?}", mv)

        //     partition: &'a Vec<PartId>,
        // cweights: &'a Vec<Vec<W>>,
        // // Targetor specific parameter
        // nb_intervals: &'a Vec<T>,
        // deltas: &'a Vec<W>,
        // parts_target_loads: &'a Vec<Vec<W>>,

        // // parts_load_per_crit: Vec<W>,
        // parts_imbalances_per_crit: &'a Vec<Vec<isize>>,

        // let parts_imbalances_per_crit = Targetor::<usize, f64>::compute_partition_imbalances(
        //     &partition,
        //     &instance.cweights,
        //     &parts_target_loads,
        // );

        // assert_eq!(
        //     vec![vec![2.5, 0.], vec![-2.5, 0.]],
        //     parts_imbalances_per_crit
        // );

        // let (criterion, part) = Targetor::<usize, f64>::process_imbalance(
        //     &partition,
        //     &instance.cweights,
        //     &parts_target_loads,
        // );

        // println!("test {:?}", (criterion, part));

        //
        //
        //
        //
        // assert!(false);
        // partition: &'a Vec<PartId>,
        // cweights: &'a Vec<Vec<W>>,
        // parts_target_loads: &'a Vec<Vec<W>>,

        // let algorithm: Targetor<,> = Targetor::new(
        //     &partition,
        //      // partition: &'a Vec<PartId>,
        //                // cweights: &'a Vec<Vec<W>>,
        //                // // Targetor specific parameter
        //                // nb_intervals: &'a Vec<T>,
        //                // deltas: &'a Vec<W>,
        //                // parts_target_loads: &'a Vec<Vec<W>>,
        //                // // parts_load_per_crit: Vec<W>,
        //                // parts_imbalances_per_crit: &'a Vec<Vec<isize>>,
        // );

        // // Split with steps 1.0 on the first criterion and 0.5 on the second one.
        // let rbh = RegularBoxHandler::new(&instance.cweights, instance.nb_intervals);
        // let mut expected_box_indices: Vec<Vec<i16>> = Vec::with_capacity(3);
        // expected_box_indices.extend([vec![0, 2], vec![1, 0], vec![3, 2]]);

        // expected_box_indices
        //     .iter()
        //     .zip(instance.cweights)
        //     .for_each(|(box_indices, cweight)| {
        //         let values = rbh.box_indices(&cweight);
        //         assert!(
        //             box_indices
        //                 .iter()
        //                 .zip(values.indices.iter())
        //                 .all(|(expected_val, computed_val)| expected_val == computed_val),
        //             "Indices are not equal {:?}, {:?} ",
        //             box_indices,
        //             values
        //         );
        //     });
    }
}

// fn main() {

// }
