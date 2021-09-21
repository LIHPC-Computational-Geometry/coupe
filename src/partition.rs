//! Utilities to manipulate partitions.

use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use num::{Num, Signed};
use snowflake::ProcessUniqueId;
use sprs::CsMatView;

use std::cmp::PartialOrd;
use std::iter::Sum;

use crate::geometry::Mbr;
use crate::PointND;

/// Represents a partition.
///
/// This struct is usually created by a partitioning algorithm. It internally uses unique IDs to represent a partition of a set of points and weights.
/// The `Partition` object exposes a convenient interface to manipulate a partition:
///  - Iterate over parts
///  - Analyze quality (e.g. weight imbalance)
///
/// # Example
///
/// ```rust
/// use coupe::Point2D;
/// use coupe::partition::Partition;
/// use approx::*;
///
/// let id1 = coupe::ProcessUniqueId::new();
/// let id2 = coupe::ProcessUniqueId::new();
/// let id3 = coupe::ProcessUniqueId::new();
///
/// let ids = vec![id1, id2, id3, id1, id2, id3, id1, id2, id3];
/// let weights = vec![1.; 9];
/// let points = vec![
///     Point2D::new(2., 2.),
///     Point2D::new(3., 3.),
///     Point2D::new(7., 7.),
///     Point2D::new(8., 8.),
///     Point2D::new(9., 9.),
///     Point2D::new(5., 5.),
///     Point2D::new(1., 1.),
///     Point2D::new(6., 6.),
///     Point2D::new(4., 4.),
/// ];
///
/// let partition = Partition::from_ids(&points, &weights, ids);
///
/// // iterate over each part corresponding to a unique ID
/// for part in partition.parts() {
///     assert_ulps_eq!(part.total_weight(), 3.)
/// }
/// ```
#[derive(Clone)]
pub struct Partition<'a, P, W> {
    points: &'a [P],
    weights: &'a [W],
    ids: Vec<ProcessUniqueId>,
}

impl<'a, P, W> Partition<'a, P, W> {
    /// Constructs a default partition from a set of points and weights and initialize it with default IDs (with a single part).
    ///
    /// # Panics
    ///
    /// Panics if `points` and `weights` have different lenghts.
    pub fn new(points: &'a [P], weights: &'a [W]) -> Self {
        if points.len() != weights.len() {
            panic!(
                "Cannot initialize a partition with points and weights of different sizes. Found {} points and {} weights",
                points.len(), weights.len()
            );
        }

        // allocate a new vector
        let dummy_id = ProcessUniqueId::new();
        let ids = vec![dummy_id; points.len()];

        Self {
            points,
            weights,
            ids,
        }
    }

    /// Constructs a partition from a set of points, weights and IDs.
    ///
    /// # Panics
    ///
    /// Panics if `points`, `weights` and `ids` do not have the same length.
    pub fn from_ids(points: &'a [P], weights: &'a [W], ids: Vec<ProcessUniqueId>) -> Self {
        if points.len() != weights.len() {
            panic!(
                "Cannot initialize a partition with points and weights of different sizes. Found {} points and {} weights",
                points.len(), weights.len()
            );
        }

        if points.len() != ids.len() {
            panic!(
                "Cannot initialize a partition with points and ids of different sizes. Found {} points and {} ids",
                points.len(), ids.len()
            );
        }

        Self {
            points,
            weights,
            ids,
        }
    }

    /// Consumes the partition, returning the internal array of ids representing the partition.
    pub fn into_ids(self) -> Vec<ProcessUniqueId> {
        self.ids
    }

    /// Consumes the partition, returning the internal components representing the partition
    ///
    /// # Example
    ///
    /// ```rust
    /// use coupe::partition::Partition;
    /// use coupe::Point2D;
    ///
    /// fn consume_partition(partition: Partition<Point2D, f64>) {
    ///     let (_points, _weights, mut _ids) = partition.into_raw();
    ///     // do something with the fields
    /// }
    /// ```
    pub fn into_raw(self) -> (&'a [P], &'a [W], Vec<ProcessUniqueId>) {
        (self.points, self.weights, self.ids)
    }

    pub fn as_raw(&self) -> (&'a [P], &'a [W], &[ProcessUniqueId]) {
        (self.points, self.weights, &self.ids)
    }

    pub fn as_raw_mut(&mut self) -> (&'a [P], &'a [W], &mut [ProcessUniqueId]) {
        (self.points, self.weights, &mut self.ids)
    }

    /// Returns a slice of the IDs used to represent the partition.
    pub fn ids(&self) -> &[ProcessUniqueId] {
        &self.ids
    }

    /// Returns of mutable slice of the IDs used to represent the partition.
    pub fn ids_mut(&mut self) -> &mut [ProcessUniqueId] {
        &mut self.ids
    }

    /// Returns a slice of the points used for the partition.
    pub fn points(&self) -> &[P] {
        self.points
    }

    /// Returns a slice of the weights used for the partition.
    pub fn weights(&self) -> &[W] {
        self.weights
    }

    /// Returns an iterator over each part of the partition.
    ///
    /// Each part is a set of points and weights that share the same ID.
    pub fn parts(&'a self) -> impl Iterator<Item = Part<'a, P, W>> {
        let indices = (0..self.points.len()).collect::<Vec<_>>();
        self.ids.iter().unique().map(move |id| {
            let indices = indices
                .iter()
                .filter(|idx| self.ids[**idx] == *id)
                .cloned()
                .collect::<Vec<_>>();
            Part::<'a, P, W> {
                partition: self,
                indices,
            }
        })
    }

    /// Computes the maximum imbalance of the partition.
    /// It is defined as the maximum difference between the weights of two different parts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use coupe::partition::Partition;
    ///
    /// let dummy_point = coupe::Point2D::new(0., 0.);
    /// let points = [dummy_point; 5];
    ///
    /// let weights = [1, 2, 3, 4, 5];
    ///
    /// let id1 = coupe::ProcessUniqueId::new();
    /// let id2 = coupe::ProcessUniqueId::new();
    /// let id3 = coupe::ProcessUniqueId::new();
    /// let ids = vec![id1, id2, id2, id2, id3];
    ///
    /// let partition = Partition::from_ids(&points[..], &weights[..], ids);
    ///
    /// let max_imbalance = partition.max_imbalance();
    /// assert_eq!(max_imbalance, 8);
    /// ```
    pub fn max_imbalance(&'a self) -> W
    where
        W: Num + PartialOrd + Signed + Sum + Clone,
    {
        let total_weights = self
            .parts()
            .map(|part| part.total_weight())
            .collect::<Vec<_>>();
        total_weights
            .iter()
            .flat_map(|w1| {
                total_weights
                    .iter()
                    .map(move |w2| (w1.clone() - w2.clone()).abs())
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            // if the partition is empty, then there is the imbalance is null
            .unwrap_or_else(W::zero)
    }

    /// Computes the relative imbalance of the partition.
    /// It is defined as follows: `relative_imbalance = maximum_imbalance / total_weight_in_partition`.
    /// It expresses the imbalance in terms of percentage of the total weight. It may be more meaningful than raw numbers in some cases.
    ///
    /// # Example
    ///
    /// ```rust
    /// use coupe::partition::Partition;
    ///
    /// let dummy_point = coupe::Point2D::new(0., 0.);
    /// let points = [dummy_point; 5];
    ///
    /// let weights = [1000, 1000, 1000, 1000, 6000];
    ///
    /// let id1 = coupe::ProcessUniqueId::new();
    /// let id2 = coupe::ProcessUniqueId::new();
    /// let id3 = coupe::ProcessUniqueId::new();
    /// let ids = vec![id1, id2, id2, id2, id3];
    ///
    /// let partition = Partition::from_ids(&points[..], &weights[..], ids);
    ///
    /// let relative_imbalance = partition.relative_imbalance();
    /// assert_eq!(relative_imbalance, 0.5); // = 50% of total weight
    /// ```
    pub fn relative_imbalance(&'a self) -> f64
    where
        W: Num + PartialOrd + Signed + Sum + Clone,
        f64: std::convert::From<W>,
    {
        let total_weights = self
            .parts()
            .map(|part| part.total_weight())
            .collect::<Vec<_>>();
        let max_imbalance = total_weights
            .iter()
            .flat_map(|w1| {
                total_weights
                    .iter()
                    .map(move |w2| (w1.clone() - w2.clone()).abs())
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or_else(W::zero);

        f64::from(max_imbalance) / f64::from(total_weights.into_iter().sum())
    }

    /// Merge two parts of the partition (identified by unique id).
    ///
    /// If either `id1` or `id2` is not contained in the partition, does nothing.
    pub fn merge_parts(&mut self, id1: ProcessUniqueId, id2: ProcessUniqueId) {
        for id in self.ids.iter_mut() {
            if *id == id2 {
                *id = id1;
            }
        }
    }

    /// Construct pairs of adjacent parts.
    /// Two parts are adjacent if there exist one element in the first
    /// part and one element in the second part that are linked together
    /// in the adjacency graph.
    pub fn adjacent_parts(
        &'a self,
        adjacency: CsMatView<f64>,
    ) -> Vec<(Part<'a, P, W>, Part<'a, P, W>)> {
        // build parts adjacency graph
        let parts = self.parts().collect::<Vec<_>>();

        let mut adjacent_parts = Vec::new();

        for (i, p) in parts.iter().enumerate() {
            for (j, q) in parts.iter().enumerate() {
                if j < i
                    && adjacency
                        .iter()
                        .any(|(_, (i, j))| p.indices().contains(&i) && q.indices().contains(&j))
                {
                    adjacent_parts.push((p.clone(), q.clone()))
                }
            }
        }

        adjacent_parts
    }

    /// Swaps the parts of two elements
    ///
    /// Panics if `idx1` or `idx2` is out of range
    pub fn swap(&mut self, idx1: usize, idx2: usize) {
        self.ids.swap(idx1, idx2);
    }
}

/// Represent a part of a partition.
///
/// This struct is not meaningful on its own, and is usually constructed by
/// [iterating over a partition](struct.Partition.html#method.parts).
pub struct Part<'a, P, W> {
    partition: &'a Partition<'a, P, W>,
    indices: Vec<usize>,
}

// Custom Clone impl since we juste clone the indices and not the partition itself
impl<'a, P, W> Clone for Part<'a, P, W> {
    fn clone(&self) -> Self {
        Self {
            partition: self.partition,
            indices: self.indices.clone(),
        }
    }
}

impl<'a, P, W> Part<'a, P, W> {
    /// Computes the total weight (i.e. the sum of all the weights) of the part.
    pub fn total_weight(&self) -> W
    where
        W: Sum + Clone,
    {
        self.indices
            .iter()
            .map(|idx| &self.partition.weights[*idx])
            .cloned()
            .sum()
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn into_indices(self) -> Vec<usize> {
        self.indices
    }

    pub fn points(&self) -> &'a [P] {
        self.partition.points()
    }

    pub fn weights(&self) -> &'a [W] {
        self.partition.weights()
    }

    /// Iterate over the points and weights of the part.
    ///
    /// # Example
    ///
    /// ```rust
    /// use coupe::Point2D;
    /// use coupe::partition::Partition;
    ///
    /// fn iterate(partition: &Partition<Point2D, f64>) {
    ///     // iterate over each part
    ///     for part in partition.parts() {
    ///         //iterate over each point/weight of the current part
    ///         for (p, w) in part.iter() {
    ///             // do something with p, w
    ///         }
    ///     }
    /// }
    /// ```
    pub fn iter(&self) -> PartIter<P, W> {
        PartIter {
            partition: self.partition,
            indices: &self.indices,
        }
    }
}

impl<'a, W, const D: usize> Part<'a, PointND<D>, W> {
    /// Computes the aspect ratio of a part. It is defined as the aspect ratio of a minimal bounding rectangle
    /// of the set of points contained in the part.
    ///
    /// # Example
    ///
    /// ```rust
    /// use coupe::Point2D;
    /// use coupe::partition::Partition;
    /// use approx::*;
    ///
    /// let points = [
    ///     Point2D::new(0., 0.),
    ///     Point2D::new(6., 0.),
    ///     Point2D::new(0., 2.),
    ///     Point2D::new(6., 2.),
    /// ];
    /// let weights = [1.; 4];
    ///
    /// let partition = Partition::new(&points[..], &weights[..]); // initialized with a single part
    ///
    /// // only 1 iteration
    /// for part in partition.parts() {
    ///     assert_ulps_eq!(part.aspect_ratio(), 3.);
    /// }
    /// ```
    pub fn aspect_ratio(&self) -> f64
    where
        Const<D>: DimSub<Const<1>>,
        DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
            + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
    {
        if self.indices.len() <= 2 {
            panic!("Cannot compute the aspect ratio of a part of less than 2 points");
        }
        let points = self
            .indices
            .iter()
            .map(|&idx| self.partition.points()[idx])
            .collect::<Vec<_>>();
        Mbr::from_points(&points).aspect_ratio()
    }
}

/// An iterator over points and weights of a part.
///
/// Is struct is usually created when [iterating over a part](struct.Part.html#method.iter).
pub struct PartIter<'a, P, W> {
    partition: &'a Partition<'a, P, W>,
    indices: &'a [usize],
}

impl<'a, P, W> Iterator for PartIter<'a, P, W> {
    type Item = (&'a P, &'a W);
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.get(0).map(|idx| {
            self.indices = &self.indices[1..];
            (&self.partition.points[*idx], &self.partition.weights[*idx])
        })
    }
}
