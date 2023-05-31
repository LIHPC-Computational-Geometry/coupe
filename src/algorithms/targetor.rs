use itertools::Itertools;
use std::cmp::{self, Ordering, PartialOrd};
use std::collections::BTreeMap;
// use std::cmp::PartialOrd;
use num_traits::{FromPrimitive, PrimInt, ToPrimitive, Zero};
use std::ops::{Add, Index, Sub};

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
    Sized + Copy + Zero + PartialOrd + Clone + Sub<Output = Self> + ToPrimitive
{
    fn try_into_positive(self) -> Result<Self, NonPositiveError>;
    fn positive_or(self) -> Option<Self>;
}

#[derive(Debug)]
pub struct NonPositiveError;

// For the moment we only implement this for i32 values
impl PositiveWeight for i32 {
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

//TODO:Update find_valid_move signature (refs and/or traits needs)
trait BoxHandler<'a, T: PositiveInteger, W: PositiveWeight> {
    fn box_indices(&self, cweight: impl IntoIterator<Item = W>) -> BoxIndices<T>;
    fn find_valid_move<CC, CP, CW>(
        &self,
        origin: &'a BoxIndices<T>,
        part_source: PartId,
        partition_imbalances: CC,
        partition: &'a CP,
        cweights: CC,
    ) -> Option<CWeightMove>
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: IntoIterator<Item = PartId> + Clone + std::ops::Index<usize, Output = PartId>,
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
    // max_weights: Vec<W>,
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
    fn new<C, I>(cweights: C, nb_intervals: impl IntoIterator<Item = T> + Clone) -> Self
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

        let deltas = Self::process_deltas(min_values.clone(), max_values, nb_intervals.clone());

        let mut res = Self {
            min_weights: min_values,
            // max_weights: max_weights,
            // nb_intervals: nb_intervals.into(),
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
                    Some(val) => cmp::min(
                        T::from((diff.to_f64().unwrap() / delta).floor()).unwrap(),
                        T::from(*nb_interval - T::from(1).unwrap()).unwrap(),
                    ),
                    None => T::zero(),
                })
                .collect(),
        );

        res
    }

    //TODO:Create some struct encapsulating partition/solution state
    fn find_valid_move<CC, CP, CW>(
        &self,
        origin: &'a BoxIndices<T>,
        part_source: PartId,
        partition_imbalances: CC,
        partition: &'a CP,
        cweights: CC,
    ) -> Option<CWeightMove>
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: IntoIterator<Item = PartId> + Clone + std::ops::Index<usize, Output = PartId>,
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

pub struct TargetorWIP {}

#[cfg(test)]
mod tests {

    use super::*;

    struct Instance {
        pub cweights: Vec<Vec<i32>>,
        pub nb_intervals: Vec<i32>,
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
        let mut expected_box_indices: Vec<Vec<i32>> = Vec::with_capacity(3);
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
                // let box_indices = rbh.box_indices(cweight);
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
}
