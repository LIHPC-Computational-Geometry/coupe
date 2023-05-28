use itertools::Itertools;
use std::cmp::{self, min};
use std::collections::BTreeMap;
use std::slice::SliceIndex;

use num_traits::{PrimInt, ToPrimitive, Zero};
use std::ops::{Add, Sub};

type CWeightId = usize;
type PartId = usize;
type CriterionId = usize;
type CWeightMove = (CWeightId, PartId);

pub trait PositiveWeight:
    Sized + Copy + Zero + PartialOrd + Clone + Sub<Output = Self> + ToPrimitive
{
    fn try_into_positive(self) -> Result<Self, NonPositiveError>;
    fn positive_or(self) -> Option<Self>;
}

#[derive(Debug)]
struct NonPositiveError;

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
trait BoxHandler<'a, T: PositiveInteger, W: PositiveWeight> {
    // fn new(cweights: &Vec<Vec<W>>, nb_intervals: Vec<T>) -> Box<dyn BoxHandler<T, W>>;
    // fn new(cweights: &Vec<Vec<W>>, nb_intervals: Vec<T>) -> Box<Self>;
    fn box_indices(&self, cweight: impl IntoIterator<Item = W>) -> BoxIndices<T>;
    // fn valid_cweight_from_indices(&self, box_indices: &'a BoxIndices<T>,
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
where
    f64: From<T>,
    f64: From<W>,
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
                (max_val - min_val).to_f64().unwrap() / f64::from(nb_interval)
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

impl<T, W> RegularBoxHandler<T, W>
where
    T: PositiveInteger,
    W: PositiveWeight,
{
    fn new<C, I>(cweights: C, nb_intervals: impl IntoIterator<Item = T>) -> Self
    where
        C: IntoIterator<Item = I>,
        I: IntoIterator<Item = W>,
        f64: From<T>,
        f64: From<W>,
    {
        let boxes: BTreeMap<BoxIndices<T>, Vec<CWeightId>> = BTreeMap::new();
        let mut cweights_iter = cweights.into_iter();
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

        let deltas = Self::process_deltas(min_values, max_values, nb_intervals);

        let mut res = Self {
            min_weights: min_values,
            // max_weights: max_weights,
            // nb_intervals: nb_intervals.into(),
            nb_intervals: nb_intervals.into_iter().collect(),

            deltas: deltas,
            boxes: boxes,
        };

        cweights_iter = cweights.into_iter();
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
        let nb_criteria: usize = self.min_weights.len();

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
                cweights: vec![vec![0, 2], vec![1, 0], vec![4, 1]],
                nb_intervals: vec![4, 3],
            };

            out
        }
    }

    #[test]
    fn check_regular_box_handler() {
        let instance = Instance::create_instance();

        // Split with steps 1.0 on the first criterion and 0.5 on the second one.
        let rbh = RegularBoxHandler::new(instance.cweights, instance.nb_intervals);
        let mut expected_box_indices: Vec<Vec<i32>> = Vec::with_capacity(3);
        expected_box_indices.extend([vec![0, 2], vec![1, 0], vec![3, 2]]);

        expected_box_indices
            .iter()
            .zip(instance.cweights)
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
    }
}