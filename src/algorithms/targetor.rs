use crate::Error;
use itertools::Itertools;
use num_traits::{FromPrimitive, PrimInt, ToPrimitive, Zero};
use std::cmp::{self, Ordering, PartialOrd};
use std::collections::BTreeMap;
// use std::error::Error;
use std::fmt::Debug;
use std::ops::IndexMut;
use std::ops::{Add, AddAssign, Sub, SubAssign};

type CWeightId = usize;
type PartId = usize;
type CriterionId = usize;
type CWeightMove = (CWeightId, PartId);

pub trait GainValue:
    Copy
    + std::fmt::Debug
    + FromPrimitive
    + ToPrimitive
    + Zero
    + Add<Output = Self>
    + Sub<Output = Self>
    + PartialOrd
{
}

impl<T> GainValue for T
where
    Self: Copy + std::fmt::Debug,
    Self: FromPrimitive + ToPrimitive + Zero,
    Self: Add<Output = Self>,
    Self: Sub<Output = Self>,
    Self: PartialOrd,
{
}

pub trait PositiveWeight:
    Sized
    + Copy
    + Zero
    + PartialOrd
    + Clone
    + Sub<Output = Self>
    + ToPrimitive
    + Debug
    + SubAssign
    + AddAssign
{
    fn try_into_positive(self) -> Result<Self, NonPositiveError>;
    fn positive_or(self) -> Option<Self>;
}

#[derive(Debug, Copy, Clone)]
pub struct NonPositiveError;
// #[derive(Clone, Debug)]
// pub enum TargetorFailure {
//     NonPositiveError,
//     WrongData(String),
// }
// #[derive(Clone, Debug)]
// pub struct TargetorError {
//     error: TargetorFailure,
//     description: String,
// }

// impl TargetorError {
//     fn new(error: TargetorFailure) -> Self {
//         TargetorError {
//             description: error.to_string(),
//             error,
//         }
//     }
// }

// impl fmt::Display for TargetorError {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "{}", self.error)
//     }
// }

// impl Error for TargetorError {
//     fn description(&self) -> &str {
//         &self.description
//     }
// }

// For the moment we only implement this for i64 values
impl PositiveWeight for i64 {
    fn try_into_positive(self) -> Result<Self, NonPositiveError> {
        if self >= 0 {
            Ok(self)
        } else {
            Err(NonPositiveError)
        }
    }

    fn positive_or(self) -> Option<Self> {
        if self >= 0 {
            Some(self)
        } else {
            None
        }
    }
}

// // Trait alias for values accepted as indices for the solution space discretization.
pub trait PositiveInteger: PrimInt + Zero + Add<Self, Output = Self> + Debug {}

impl<T: PrimInt + Zero + Add<Output = Self> + Debug> PositiveInteger for T {}

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

impl<'a, T: PositiveInteger> SearchStrat<'a, T> for NeighborSearchStrat<T> {
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
        for criterion in 0..nb_criteria {
            let min_rng = -cmp::min(dist, left_bounds[criterion]).to_isize().unwrap();
            let max_rng = cmp::min(dist, right_bounds[criterion]).to_isize().unwrap();
            let rng = min_rng..=max_rng;
            rngs.push(rng);
        }

        let indices_generator = rngs
            .into_iter()
            .multi_cartesian_product()
            .filter(move |indices| {
                indices.iter().map(|i| i.abs()).sum::<isize>() == dist.to_isize().unwrap()
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

//TODO:Update find_valid_move signature (refs and/or traits needs)
trait BoxHandler<'a, T: PositiveInteger, W: PositiveWeight> {
    fn box_indices(&self, cweight: impl IntoIterator<Item = W>) -> BoxIndices<T>;
    fn find_valid_move<CC, CP, CW>(
        &self,
        origin: &'a BoxIndices<T>,
        part_source: PartId,
        partition_imbalances: Vec<Vec<W>>,
        partition: &'a CP,
        cweights: CC,
    ) -> Option<CWeightMove>
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: std::ops::Index<usize, Output = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>;
}

// // Trait alias for handling regular delta of the solution space discretization.
trait RegularDeltaHandler<'a, T: PositiveInteger, W: PositiveWeight> {
    fn process_deltas(
        min_weights: impl IntoIterator<Item = W>,
        max_weights: impl IntoIterator<Item = W>,
        nb_intervals: impl IntoIterator<Item = T>,
    ) -> Vec<f64>;
}

impl<'a, T: PositiveInteger, W: PositiveWeight> RegularDeltaHandler<'a, T, W>
    for RegularBoxHandler<T, W>
{
    // Compute the regular delta associated with each criterion
    fn process_deltas(
        min_weights: impl IntoIterator<Item = W>,
        max_weights: impl IntoIterator<Item = W>,
        nb_intervals: impl IntoIterator<Item = T>,
    ) -> Vec<f64> {
        let res = min_weights
            .into_iter()
            .zip(max_weights.into_iter())
            .zip(nb_intervals.into_iter())
            .map(|((min_val, max_val), nb_interval)| {
                (max_val - min_val).to_f64().unwrap() / nb_interval.to_f64().unwrap()
            })
            .collect();

        res
    }
}

struct RegularBoxHandler<T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    // Values related to the instance to be improved
    min_weights: Vec<W>,
    // Parameters related to the discretization of the solution space
    nb_intervals: Vec<T>,
    deltas: Vec<f64>,
    // Mapping used to search moves of specific gain from a discretization of the cweight space
    pub boxes: BTreeMap<BoxIndices<T>, Vec<CWeightId>>,
}

//FIXME:Refact this code to avoid clone calls.
impl<T, W> RegularBoxHandler<T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    // pub fn setup_from_instance<C, I>(&mut self, cweights: C)
    // where
    //     C: IntoIterator<Item = I> + Clone,
    //     I: IntoIterator<Item = W>,
    // {
    //     let mut cweights_iter = cweights.clone().into_iter();
    //     let first_cweight = cweights_iter.next().unwrap();
    //     let mut min_values = first_cweight.into_iter().collect::<Vec<_>>();
    //     let mut max_values = min_values.clone();

    //     // Set up min_weights
    //     for cweight in cweights_iter {
    //         cweight
    //             .into_iter()
    //             .zip(min_values.iter_mut())
    //             .zip(max_values.iter_mut())
    //             .for_each(|((current_val, min_val), max_val)| {
    //                 if current_val < *min_val {
    //                     *min_val = current_val.clone();
    //                 }
    //                 if current_val > *max_val {
    //                     *max_val = current_val.clone();
    //                 }
    //             })
    //     }
    //     self.min_weights = min_values.clone();

    //     // Set up regular deltas
    //     let deltas = Self::process_deltas(
    //         min_values.clone(),
    //         max_values.clone(),
    //         self.nb_intervals.clone(),
    //     );
    //     self.deltas = deltas;

    //     // Set up boxes mapping
    //     let boxes: BTreeMap<BoxIndices<T>, Vec<CWeightId>> = BTreeMap::new();
    //     cweights_iter = cweights.clone().into_iter();
    //     cweights_iter.enumerate().for_each(|(cweight_id, cweight)| {
    //         let indices: BoxIndices<T> = Self::box_indices(self, cweight);
    //         match self.boxes.get_mut(&indices) {
    //             Some(vect_box_indices) => vect_box_indices.push(cweight_id),
    //             None => {
    //                 let vect_box_indices: Vec<usize> = vec![cweight_id];
    //                 self.boxes.insert(indices, vect_box_indices);
    //             }
    //         }
    //     });
    //     self.boxes = boxes;
    // }

    pub fn new<C, I>(cweights: C, nb_intervals: impl IntoIterator<Item = T> + Clone) -> Self
    where
        C: IntoIterator<Item = I> + Clone,
        I: IntoIterator<Item = W>,
    {
        let boxes: BTreeMap<BoxIndices<T>, Vec<CWeightId>> = BTreeMap::new();
        let mut cweights_iter = cweights.clone().into_iter();
        let first_cweight = cweights_iter.next().unwrap();
        let mut min_values = first_cweight.into_iter().collect::<Vec<_>>();
        let mut max_values = min_values.clone();

        for cweight in cweights_iter {
            cweight
                .into_iter()
                .zip(min_values.iter_mut())
                .zip(max_values.iter_mut())
                .for_each(|((current_val, min_val), max_val)| {
                    if current_val < *min_val {
                        *min_val = current_val.clone();
                    }
                    if current_val > *max_val {
                        *max_val = current_val.clone();
                    }
                })
        }

        let deltas =
            Self::process_deltas(min_values.clone(), max_values.clone(), nb_intervals.clone());

        let mut res = Self {
            min_weights: min_values,
            nb_intervals: nb_intervals.into_iter().collect(),
            deltas: deltas,
            boxes: boxes,
        };

        cweights_iter = cweights.clone().into_iter();
        cweights_iter.enumerate().for_each(|(cweight_id, cweight)| {
            let indices: BoxIndices<T> = Self::box_indices(&res, cweight);
            match res.boxes.get_mut(&indices) {
                Some(vect_box_indices) => vect_box_indices.push(cweight_id),
                None => {
                    let vect_box_indices: Vec<usize> = vec![cweight_id];
                    res.boxes.insert(indices, vect_box_indices);
                }
            }
        });

        res
    }
}

impl<'a, T, W> Debug for RegularBoxHandler<T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Regular Box Handler with min_weights {:?}, nb_intervals {:?} and deltas {:?}",
            self.min_weights, self.nb_intervals, self.deltas
        )
    }
}

impl<'a, T: PositiveInteger, W: PositiveWeight> BoxHandler<'a, T, W> for RegularBoxHandler<T, W>
where
    W: Sub<W, Output = W> + Zero + ToPrimitive,
{
    fn box_indices(&self, cweight: impl IntoIterator<Item = W>) -> BoxIndices<T> {
        let res = BoxIndices::new(
            cweight
                .into_iter()
                .zip(self.min_weights.iter())
                .map(|(val, min)| val - *min)
                .zip(self.deltas.iter())
                .zip(self.nb_intervals.iter())
                .map(|((diff, delta), nb_interval)| match diff.positive_or() {
                    Some(_) => cmp::min(
                        T::from((diff.to_f64().unwrap() / delta).floor()).unwrap(),
                        T::from(*nb_interval - T::from(1).unwrap()).unwrap(),
                    ),
                    None => T::zero(),
                })
                .collect(),
        );

        res
    }

    //FIXME:Allow partition imbalance to be composed of float values while cweights are integers
    //TODO:Create some struct encapsulating partition/solution state
    fn find_valid_move<CC, CP, CW>(
        &self,
        origin: &'a BoxIndices<T>,
        part_source: PartId,
        partition_imbalances: Vec<Vec<W>>,
        partition: &'a CP,
        cweights: CC,
    ) -> Option<CWeightMove>
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: std::ops::Index<usize, Output = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        let imbalances_iter = partition_imbalances.clone().into_iter();
        let max_imbalances = imbalances_iter
            .map(|criterion_imbalances| {
                let mut criterion_imbalances_iter = criterion_imbalances.into_iter();
                let first_part_imbalance = criterion_imbalances_iter.next().unwrap();
                match first_part_imbalance.positive_or() {
                    Some(val) => val,
                    None => criterion_imbalances_iter.next().unwrap(),
                }
            })
            .collect::<Vec<_>>();
        // let partition_imbalance: W = *max_imbalances.iter().max().unwrap();
        let partition_imbalance = max_imbalances
            .iter()
            .fold(None, |acc, &imbalance| match acc {
                None => Some(imbalance),
                Some(max) => {
                    let comparison = imbalance.partial_cmp(&max);
                    match comparison {
                        Some(Ordering::Greater) => Some(imbalance),
                        Some(Ordering::Less) | Some(Ordering::Equal) => Some(max),
                        None => acc,
                    }
                }
            })
            .unwrap();

        let strict_positive_gain = |id| {
            max_imbalances
                .iter()
                .zip(cweights[id].clone().into_iter())
                .zip(partition_imbalances.clone().into_iter())
                // .map(|((max, cweight), criterion_imbalances)| {
                //     *max == partition_imbalance
                //         && cweight >= (partition_imbalance + partition_imbalance)
                //         || *max != partition_imbalance
                //             && cweight
                //                 >= partition_imbalance - criterion_imbalances[1 - part_source]
                // })
                // .any(|invalid_gain| !invalid_gain)
                .map(|((max, cweight), criterion_imbalances)| {
                    *max == partition_imbalance
                        && cweight < (partition_imbalance + partition_imbalance)
                        || *max != partition_imbalance
                            && cweight
                                < (partition_imbalance - criterion_imbalances[1 - part_source])
                })
                .all(|valid_gain| valid_gain)
        };

        let candidates = self.boxes.get(&origin);
        // println!("The candidates are {:?}", candidates);
        let candidate_move = candidates
            .unwrap_or(&vec![])
            .iter()
            // Filter moves
            .filter(|id| partition[**id] == part_source)
            // Filter settled cweights, i.e. cweights whose move should leave
            // to a higher partition imbalance
            .find(|id| strict_positive_gain(**id))
            .map(|id| (*id, 1 - part_source))
            .clone();

        candidate_move
    }
}

trait Repartitioning<'a, T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    // TODO: Add a builder for generic Repartitioning
    fn optimize<CC, CP, CW>(&mut self, partition: &'a mut CP, cweights: CC)
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: IntoIterator<Item = PartId>
            + std::ops::Index<usize, Output = PartId>
            + IndexMut<usize>
            + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>;
}

trait PartitionImbalanceHandler<'a, T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    fn compute_imbalances<CC, CP, CW>(&self, partition: &'a CP, cweights: CC) -> Vec<Vec<W>>
    where
        CC: IntoIterator<Item = CW> + Clone,
        CP: IntoIterator<Item = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone;

    // Return a couple composed of one of the most imbalanced criterion and
    // the part from which a weight should me moved.
    // FIXME:Add support to handle multiple criterion having the most imbalanced value
    fn process_imbalance<CC, CP, CW>(
        &self,
        partition: &'a CP,
        cweights: CC,
    ) -> (CriterionId, PartId)
    where
        CC: IntoIterator<Item = CW> + Clone,
        CP: IntoIterator<Item = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>;
}

#[derive(Debug)]
pub struct TargetorWIP<T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    // Instance data
    nb_intervals: Vec<T>,
    parts_target_loads: Vec<Vec<W>>,

    // // Partition state
    // partition: Vec<PartId>,
    // box_handler: Box<dyn BoxHandler<'a, T, W>>,
    box_handler: Option<RegularBoxHandler<T, W>>,
}

impl<'a, T: PositiveInteger, W: PositiveWeight> PartitionImbalanceHandler<'a, T, W>
    for TargetorWIP<T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    fn compute_imbalances<CC, CP, CW>(&self, partition: &'a CP, cweights: CC) -> Vec<Vec<W>>
    where
        CC: IntoIterator<Item = CW> + Clone,
        CP: IntoIterator<Item = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone,
    {
        let nb_criterion = self.parts_target_loads.len();
        let mut res = vec![vec![W::zero(); 2]; nb_criterion];
        for criterion in 0..nb_criterion {
            for part in 0..2 {
                res[criterion][part] -= self.parts_target_loads[criterion][part];
            }
        }

        let cweights_iter = cweights.clone().into_iter();
        partition
            .clone()
            .into_iter()
            .zip(cweights_iter)
            .for_each(|(part, cweight)| {
                cweight
                    .into_iter()
                    .enumerate()
                    .for_each(|(criterion, weight)| res[criterion][part] += weight)
            });
        res
    }

    /* Function returning a couple composed of the most imbalanced criterion
    and the part from which a weight should me moved.
     */
    fn process_imbalance<CC, CP, CW>(
        &self,
        partition: &'a CP,
        cweights: CC,
    ) -> (CriterionId, PartId)
    where
        CC: IntoIterator<Item = CW> + Clone,
        CP: IntoIterator<Item = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        let imbalances = self.compute_imbalances(partition, cweights);
        let mut max_imb = W::zero();
        let mut res: (CriterionId, PartId) = (CriterionId::zero(), PartId::zero());

        imbalances
            .into_iter()
            .enumerate()
            .for_each(|(criterion, loads)| {
                let max = if loads[0] >= W::zero() {
                    loads[0]
                } else {
                    loads[1]
                };
                if max > max_imb {
                    let part = 1 - (loads[0] >= loads[1]) as usize;
                    res = (criterion, part);
                    max_imb = max;
                }
            });

        res
    }
}

impl<'a, T: PositiveInteger, W: PositiveWeight> TargetorWIP<T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    //FIXME:Allow partition imbalance to be composed of float values while cweights are integers
    pub fn new<CC, CT, CW>(
        // partition: Vec<PartId>,
        // cweights: CC,
        // Targetor specific parameter
        nb_intervals: CT,
        parts_target_loads: CC,
    ) -> Self
    where
        CC: IntoIterator<Item = CW> + Clone,
        CT: IntoIterator<Item = T> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        // let res_rbh = RegularBoxHandler::new(cweights, nb_intervals.clone());
        let res_target_loads: Vec<Vec<W>> = parts_target_loads
            .into_iter()
            .map(|criterion_target_loads| criterion_target_loads.into_iter().collect())
            .collect();

        let res = Self {
            nb_intervals: nb_intervals.into_iter().collect(),
            parts_target_loads: res_target_loads,
            box_handler: None,
        };

        res
    }

    pub fn setup_default_box_handler<CC, CW>(&mut self, cweights: CC)
    where
        CC: IntoIterator<Item = CW> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        let box_handler = RegularBoxHandler::new(cweights, self.nb_intervals.clone());
        self.box_handler = Some(box_handler);
    }
}

impl<'a, T: PositiveInteger, W: PositiveWeight> Repartitioning<'a, T, W> for TargetorWIP<T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    fn optimize<CC, CP, CW>(&mut self, partition: &'a mut CP, cweights: CC)
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: IntoIterator<Item = PartId>
            + std::ops::Index<usize, Output = PartId>
            + IndexMut<usize>
            + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        if self.box_handler.is_none() {
            self.setup_default_box_handler(cweights.clone());
        }
        let box_handler = self.box_handler.as_ref().unwrap();

        // Setup search strat
        let search_strat = NeighborSearchStrat::new(self.nb_intervals.clone());

        let mut look_for_movement = true;
        while look_for_movement {
            // Setup target part and most imbalanced criteria
            let (most_imbalanced_criterion, part_source) =
                self.process_imbalance(partition, cweights.clone());

            // Setup target gain
            let partition_imbalances = self.compute_imbalances(partition, cweights.clone());
            let nb_criteria = partition_imbalances.len();
            let target_gain: Vec<W> = partition_imbalances
                .clone()
                .into_iter()
                .map(
                    |criterion_imbalances| match criterion_imbalances[part_source].positive_or() {
                        Some(val) => val,
                        None => W::zero(),
                    },
                )
                .collect();

            let origin = box_handler.box_indices(target_gain.clone());
            let option_valid_move = box_handler.find_valid_move(
                &origin,
                part_source,
                partition_imbalances.clone(),
                partition,
                cweights.clone(),
            );

            if option_valid_move.is_some() {
                let (id_cweight, target_part) = option_valid_move.unwrap();
                partition[id_cweight] = target_part;
            } else {
                let mut increase_offset = true;
                let mut offset = 1;
                while increase_offset {
                    // println!("Increasing offset to {}", offset);
                    let iter_indices = search_strat.gen_indices(&origin, T::from(offset).unwrap());
                    if let Some(option_valid_move) = iter_indices
                        .into_iter()
                        .map(|box_indices| {
                            box_handler.find_valid_move(
                                &box_indices,
                                part_source,
                                partition_imbalances.clone(),
                                partition,
                                cweights.clone(),
                            )
                        })
                        .find(|option_valid_move| option_valid_move.is_some())
                    {
                        increase_offset = false;
                        let (id_cweight, target_part) = option_valid_move.unwrap();
                        partition[id_cweight] = target_part;
                    } else {
                        offset += 1;
                        let partition_imbalance =
                            partition_imbalances[most_imbalanced_criterion][part_source];
                        let bound_indices =
                            box_handler.box_indices(vec![partition_imbalance; nb_criteria]);
                        increase_offset = (0..nb_criteria).all(|criterion| {
                            origin.indices[criterion] - T::from(offset).unwrap() >= T::zero()
                                || origin.indices[criterion] + T::from(offset).unwrap()
                                    <= bound_indices.indices[criterion]
                        });
                        look_for_movement = increase_offset;
                    }
                }
            }
        }
    }
}

impl<'a, T: PositiveInteger, W: PositiveWeight, CC, CW> crate::Partition<CC> for TargetorWIP<T, W>
where
    CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
    CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
{
    // impl<'a, T: PositiveInteger, W: PositiveWeight, CC> crate::Partition<CC> for TargetorWIP<T, W>
    // where
    //     CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
    //     CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    // {
    type Metadata = usize;
    type Error = Error;

    // fn optimize<CC, CP, CW>(&mut self, partition: &'a mut CP, cweights: CC)
    // where
    //     CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
    //     CP: IntoIterator<Item = PartId> + Clone + std::ops::Index<usize, Output = PartId>,
    //     CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        // part_ids: &mut CP,
        cweights: CC,
    ) -> Result<Self::Metadata, Self::Error>
// where
    //     // CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
    //     CP: IntoIterator<Item = PartId> + Clone + std::ops::Index<usize, Output = PartId>,
    //     // CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        println!("Running partition");
        let partition_len = part_ids.iter().count();
        let nb_cweights = cweights.clone().into_iter().count();
        if partition_len != nb_cweights {
            return Err(Error::InputLenMismatch {
                expected: partition_len,
                actual: nb_cweights,
            });
        }

        if 1 < part_ids.iter_mut().max().map(|&mut part| part).unwrap_or(0) {
            return Err(Error::BiPartitioningOnly);
        }

        // let nb_criteria = self.parts_target_loads.len();
        // let cweights_sums =
        //     cweights
        //         .into_iter()
        //         .fold(vec![W::zero(); nb_criteria], |acc, vector| {
        //             acc.iter()
        //                 .zip(vector)
        //                 .map(|(val_0, val_1)| *val_0 + val_1)
        //                 .collect()
        //         });
        // for criterion in 0..nb_criteria {
        //     if self.parts_target_loads[criterion].iter().sum() != cweights_sums[criterion] {
        //         Err(panic!(
        //             "target loads sum does not match weight sum for criterion {}",
        //             criterion
        //         ));
        //     }
        // }

        let mut vec_partition = part_ids.to_vec();
        self.optimize(&mut vec_partition, cweights);
        for (index, value) in vec_partition.iter().enumerate() {
            part_ids[index] = *value;
        }

        return Ok(0);
    }
}

// fn partition(
//     &mut self,
//     part_ids: &mut [usize],
//     (adjacency, weights): (T, &'a [W]),
// ) -> Result<Self::Metadata, Self::Error> {
//     if part_ids.is_empty() {
//         return Ok(Metadata::default());
//     }
//     if part_ids.len() != weights.len() {
//         return Err(Error::InputLenMismatch {
//             expected: part_ids.len(),
//             actual: weights.len(),
//         });
//     }
//     if part_ids.len() != adjacency.len() {
//         return Err(Error::InputLenMismatch {
//             expected: part_ids.len(),
//             actual: adjacency.len(),
//         });
//     }
//     if 1 < *part_ids.iter().max().unwrap_or(&0) {
//         return Err(Error::BiPartitioningOnly);
//     }
//     let metadata = fiduccia_mattheyses(
//         part_ids,
//         weights,
//         adjacency,
//         self.max_passes.unwrap_or(usize::MAX),
//         self.max_moves_per_pass.unwrap_or(usize::MAX),
//         self.max_imbalance,
//         self.max_bad_move_in_a_row,
//     );
//     Ok(metadata)
// }

#[cfg(test)]
mod tests {

    use super::*;

    struct Instance {
        pub cweights: Vec<Vec<i64>>,
        pub nb_intervals: Vec<i64>,
    }

    impl Instance {
        fn create_instance() -> Self {
            let out = Self {
                cweights: vec![vec![2, 2], vec![4, 6], vec![6, 4], vec![8, 5]],
                nb_intervals: vec![3, 2],
            };

            out
        }
    }

    #[test]
    fn check_regular_box_handler() {
        let instance = Instance::create_instance();

        // Split with steps 1.0 on the first criterion and 0.5 on the second one.
        let rbh = RegularBoxHandler::new(instance.cweights.clone(), instance.nb_intervals);
        let mut expected_box_indices: Vec<Vec<i64>> = Vec::with_capacity(3);
        expected_box_indices.extend([vec![0, 0], vec![1, 1], vec![2, 1], vec![2, 1]]);

        expected_box_indices
            .iter()
            .zip(instance.cweights.clone())
            .for_each(|(box_indices, cweight)| {
                let values = rbh.box_indices(cweight);
                assert!(
                    box_indices
                        .iter()
                        .zip(values.indices.iter())
                        .all(|(expected_val, computed_val)| expected_val == computed_val),
                    "Indices are not equal {:?}, {:?} ",
                    box_indices,
                    values
                );
            });

        let part_source = 0;
        let partition = vec![part_source; instance.cweights.len()];
        let partition_imbalances = vec![vec![10, -10], vec![9, -8]];
        let space_box_indices = vec![
            BoxIndices::new(vec![0, 0]),
            BoxIndices::new(vec![1, 0]),
            BoxIndices::new(vec![2, 0]),
            BoxIndices::new(vec![0, 1]),
            BoxIndices::new(vec![1, 1]),
            BoxIndices::new(vec![2, 1]),
        ];
        let iter_box_indices = space_box_indices.iter();

        let expected_moves = vec![Some((0, 1)), None, None, None, Some((1, 1)), Some((2, 1))];
        expected_moves
            .iter()
            .zip(iter_box_indices)
            .for_each(|(expected_move, box_indices)| {
                let candidate_move = rbh.find_valid_move(
                    &box_indices,
                    part_source,
                    partition_imbalances.clone(),
                    &partition.clone(),
                    instance.cweights.clone(),
                );
                assert!(
                    *expected_move == candidate_move,
                    "Moves are not equal. Expected {:?} but returned {:?} ",
                    *expected_move,
                    candidate_move,
                );
            });
    }

    #[test]
    fn check_targetor_without_search() {
        let instance = Instance::create_instance();
        let mut partition = vec![0; instance.cweights.len()];
        let rbh = RegularBoxHandler::new(instance.cweights.clone(), instance.nb_intervals);
        let partition_target_loads = vec![vec![10, 10], vec![9, 8]];

        let mut targetor = TargetorWIP::new(
            // partition.clone(),
            rbh.nb_intervals.clone(),
            // instance.cweights.clone(),
            partition_target_loads,
        );

        targetor.optimize(&mut partition, instance.cweights.clone());
        let expected_partition_res = vec![0, 1, 1, 0];
        // assert!(
        //     targetor.partition == expected_partition_res,
        //     "Partition are not equal. Expected {:?}, returned {:?}",
        //     expected_partition_res,
        //     targetor.partition
        // );
        assert!(
            partition == expected_partition_res,
            "Partition are not equal. Expected {:?}, returned {:?}",
            expected_partition_res,
            partition
        );
    }

    #[test]
    fn check_targetor_with_search() {
        let instance = Instance::create_instance();
        let mut partition = vec![0; instance.cweights.len()];

        // Custom made data triggering exploration on the bottom right of the
        // weight space.
        let nb_intervals = vec![6, 2];
        let partition_target_loads = vec![vec![12, 8], vec![15, 2]];

        let rbh = RegularBoxHandler::new(instance.cweights.clone(), nb_intervals);

        let mut targetor = TargetorWIP::new(
            // partition.clone(),
            rbh.nb_intervals.clone(),
            // instance.cweights.clone(),
            partition_target_loads,
        );

        targetor.optimize(&mut partition, instance.cweights.clone());
        let expected_partition_res = vec![0, 0, 0, 1];
        assert!(
            partition == expected_partition_res,
            "Partition are not equal. Expected {:?}, returned {:?}",
            expected_partition_res,
            partition
        );
    }
}