//! A mesh partitioning library that implements multithreaded, composable geometric algorithms.
//!
//! # Crate Layout
//!
//! Coupe exposes each algorithm with a struct that implements a trait. There are currently two traits available:
//!
//! - [`Partitioner`] represents an algorithm that will generate a partition given a set of geometric points and weights.
//! - [`PartitionImprover`] represents an algorithm that will improve an existing partition (previously generated with a [`Partitioner`]).
//!
//! # Available algorithms
//!
//! ## Partitioner algorithms
//! - Space filling curves:
//!   + [`Z-curve`]
//!   + [`Hilbert curve`]
//! - [`Rcb`]: Recursive Coordinate Bisection
//! - [`Rib`]: Recursive Inertial Bisection
//! - [`Multi jagged`]
//!
//! ## Partition improving algorithms
//! - [`KMeans`]
//!
//! [`Partitioner`]: trait.Partitioner.html
//! [`PartitionImprover`]: trait.PartitionImprover.html
//! [`Z-curve`]: struct.ZCurve.html
//! [`Hilbert curve`]: struct.HilbertCurve.html
//! [`Rcb`]: struct.Rcb.html
//! [`Rib`]: struct.Rib.html
//! [`Multi jagged`]: struct.MultiJagged.html
//! [`KMeans`]: struct.KMeans.html

pub mod algorithms;
pub mod geometry;
pub mod partition;
pub mod topology;

#[cfg(test)]
mod tests;

// API

// SUBMODULES REEXPORT
pub use crate::geometry::{Point2D, Point3D, PointND};
pub use snowflake::ProcessUniqueId;

pub mod dimension {
    pub use nalgebra::base::dimension::*;
}

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::{DimDiff, DimSub};
use nalgebra::DefaultAllocator;
use nalgebra::DimName;
use nalgebra::U1;
use std::marker::PhantomData;

use crate::partition::*;

// Trait that allows conversions from/to different kinds of
// points views representation as partitioner inputs
// e.g. &[f64], &[PointND<D>], slice from ndarray, ...
pub trait PointsView<'a, Dim> {
    fn to_points_nd(self) -> &'a [PointND<Dim>]
    where
        Dim: DimName,
        DefaultAllocator: Allocator<f64, Dim>;
}

impl<'a, D> PointsView<'a, D> for &'a [f64] {
    fn to_points_nd(self) -> &'a [PointND<D>]
    where
        D: DimName,
        DefaultAllocator: Allocator<f64, D>,
    {
        let dim = D::dim();
        if self.len() % dim != 0 {
            panic!("error: tried to convert a &[f64] to a &[PointND<D>] with D = {}, but input slice has len {}", dim, self.len());
        }
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const PointND<D>, self.len() / dim) }
    }
}

impl<'a, D> PointsView<'a, D> for &'a [PointND<D>]
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    fn to_points_nd(self) -> &'a [PointND<D>] {
        // nothing to do
        self
    }
}

/// # A geometric partitioning algorithm.
///
/// Algorithms that implement [`Partitioner`](trait.Partitioner.html) operate on a set of geometric points and associated weights and generate
/// a partition. A partition is described by an array of ids, each unique id represents a part of the partition.
///
/// These algorithms generate a partition from scratch, as opposed to those which implement [`PartitionImprover`](trait.PartitionImprover.html), which work on
/// an existing partition to improve it.
///
/// See the [implementors](trait.Partitioner.html#implementors) for more information about the currently available algorithms.
pub trait Partitioner<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<D>, f64>;
}

/// A geometric algorithm to improve a partition.
///
/// Algorithms that implement [`PartitionImprover`](trait.PartitionImprover.html) operate on a set of geometric
/// points and associated weights to modify and improve an existing partition (typically generated by a [`Partitioner`](trait.Partitioner.html)).
///
/// See the [implementors](trait.PartitionImprover.html#implementors) for more information about the currently available algorithms.
pub trait PartitionImprover<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    fn improve_partition<'a>(
        &self,
        partition: Partition<'a, PointND<D>, f64>,
    ) -> Partition<'a, PointND<D>, f64>;
}

/// # Recursive Coordinate Bisection algorithm
///
/// Partitions a mesh based on the nodes coordinates and coresponding weights.
///
/// This is the most simple and straightforward geometric algorithm. It operates as follows for a N-dimensional set of points:
///
/// At each iteration, select a vector `n` of the canonical basis `(e_0, ..., e_{n-1})`. Then, split the set of points with an hyperplane orthogonal
/// to `n`, such that the two parts of the splits are evenly weighted. Finally, recurse by reapplying the algorithm to the two parts with an other
/// normal vector selection.
///
/// # Example
///
/// ```rust
/// use coupe::Point2D;
/// use coupe::Partitioner;
///
/// let points = vec![
///     Point2D::new(1., 1.),
///     Point2D::new(-1., 1.),
///     Point2D::new(1., -1.),
///     Point2D::new(-1., -1.),
/// ];
///
/// let weights = vec![1., 1., 1., 1.];
///
/// // generate a partition of 4 parts
/// let rcb = coupe::Rcb { num_iter: 2 };
/// let partition = rcb.partition(&points, &weights);
///
/// let ids = partition.ids();
/// for i in 0..4 {
///     for j in 0..4 {
///         if j == i {
///             continue
///         }
///         assert_ne!(ids[i], ids[j])
///     }
/// }
/// ```
pub struct Rcb<D> {
    pub num_iter: usize,
    _marker: PhantomData<D>,
}

impl<D> Rcb<D> {
    pub fn new(num_iter: usize) -> Self {
        Self {
            num_iter,
            _marker: PhantomData::<D>,
        }
    }
}

impl<D> Partitioner<D> for Rcb<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<D>, f64> {
        let points = points.to_points_nd();
        let ids = crate::algorithms::recursive_bisection::rcb(points, weights, self.num_iter);
        Partition::from_ids(points, weights, ids)
    }
}

/// # Recursive Inertial Bisection algorithm
///
/// Partitions a mesh based on the nodes coordinates and coresponding weights
///
/// This is a variant of the [`Rcb`](struct.Rcb.html) algorithm, where a basis change is performed beforehand so that
/// the first coordinate of the new basis is colinear to the inertia axis of the set of points. This has the goal
/// of producing better shaped partition than [Rcb](struct.Rcb.html).
///
/// # Example
///
/// ```rust
/// use coupe::Point2D;
/// use coupe::Partitioner;
///
/// // Here, the inertia axis is the y axis.
/// // We thus expect Rib to split horizontally first.
/// let points = vec![
///     Point2D::new(1., 10.),
///     Point2D::new(-1., 10.),
///     Point2D::new(1., -10.),
///     Point2D::new(-1., -10.),
/// ];
///
/// let weights = vec![1., 1., 1., 1.];
///
/// // generate a partition of 2 parts (1 split)
/// let rib = coupe::Rib { num_iter: 1 };
/// let partition = rib.partition(&points, &weights);
///
/// let ids = partition.ids();
///
/// // the two points at the top are in the same partition
/// assert_eq!(ids[0], ids[1]);
///
/// // the two points at the bottom are in the same partition
/// assert_eq!(ids[2], ids[3]);
///
/// // there are two different partition
/// assert_ne!(ids[1], ids[2]);
/// ```
pub struct Rib<D> {
    /// The number of iterations of the algorithm. This will yield a partition of `2^num_iter` parts.
    pub num_iter: usize,
    _marker: PhantomData<D>,
}

impl<D> Rib<D> {
    pub fn new(num_iter: usize) -> Self {
        Self {
            num_iter,
            _marker: PhantomData::<D>,
        }
    }
}

impl<D> Partitioner<D> for Rib<D>
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<D>, f64> {
        let points = points.to_points_nd();
        let ids = crate::algorithms::recursive_bisection::rib(points, weights, self.num_iter);
        Partition::from_ids(points, weights, ids)
    }
}

/// # Multi-Jagged algorithm
///
/// This algorithm is inspired by Multi-Jagged: A Scalable Parallel Spatial Partitioning Algorithm"
/// by Mehmet Deveci, Sivasankaran Rajamanickam, Karen D. Devine, Umit V. Catalyurek.
///
/// It improves over [RCB](struct.Rcb.html) by following the same idea but by creating more than two subparts
/// in each iteration which leads to decreasing recursion depth. It also allows to generate a partition
/// of any number of parts.
///
/// More precisely, given a number of parts, the algorithm will generate a "partition scheme", which describes how
/// to perform splits at each iteration, such that the total number of iteration is less than `max_iter`.
///
/// More iteration does not necessarily result in a better partition.
///
/// # Example
///
/// ```rust
/// use coupe::Point2D;
/// use coupe::Partitioner;
///
/// let points = vec![
///     Point2D::new(0., 0.),
///     Point2D::new(1., 0.),
///     Point2D::new(2., 0.),
///     Point2D::new(0., 1.),
///     Point2D::new(1., 1.),
///     Point2D::new(2., 1.),
///     Point2D::new(0., 2.),
///     Point2D::new(1., 2.),
///     Point2D::new(2., 2.),
/// ];
///
/// let weights = vec![1.; 9];
///
/// // generate a partition of 4 parts
/// let multi_jagged = coupe::MultiJagged {
///     num_partitions: 9,
///     max_iter: 4,
/// };
///
/// let partition = multi_jagged.partition(&points, &weights);
///
/// let ids = partition.ids();
///
/// for i in 0..9 {
///     for j in 0..9 {
///         if j == i {
///             continue    
///         }
///         assert_ne!(ids[i], ids[j])
///     }
/// }
/// ```
pub struct MultiJagged<D> {
    pub num_partitions: usize,
    pub max_iter: usize,
    _marker: PhantomData<D>,
}

impl<D> MultiJagged<D> {
    pub fn new(num_partitions: usize, max_iter: usize) -> Self {
        Self {
            num_partitions,
            max_iter,
            _marker: PhantomData::<D>,
        }
    }
}

impl<D> Partitioner<D> for MultiJagged<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<D>, f64> {
        let points = points.to_points_nd();
        let ids = crate::algorithms::multi_jagged::multi_jagged(
            points,
            weights,
            self.num_partitions,
            self.max_iter,
        );

        Partition::from_ids(points, weights, ids)
    }
}

/// # Z space-filling curve algorithm
///
/// The Z-curve uses space hashing to partition points. The points in the same part of a partition
/// have the same Z-hash. This hash is computed by recursively constructing a N-dimensional region tree.
///
/// # Example
///
/// ```rust
/// use coupe::Point2D;
/// use coupe::Partitioner;
///
/// let points = vec![
///     Point2D::new(0., 0.),
///     Point2D::new(1., 1.),
///     Point2D::new(0., 10.),
///     Point2D::new(1., 9.),
///     Point2D::new(9., 1.),
///     Point2D::new(10., 0.),
///     Point2D::new(10., 10.),
///     Point2D::new(9., 9.),
/// ];
///
/// let weights = vec![1.; 8];
///
/// // generate a partition of 4 parts
/// let z_curve = coupe::ZCurve {
///     num_partitions: 4,
///     order: 5,
/// };
///
/// let partition = z_curve.partition(&points, &weights);
/// let ids = partition.ids();
///
/// assert_eq!(ids[0], ids[1]);
/// assert_eq!(ids[2], ids[3]);
/// assert_eq!(ids[4], ids[5]);
/// assert_eq!(ids[6], ids[7]);
/// ```  
pub struct ZCurve<D> {
    pub num_partitions: usize,
    pub order: u32,
    _marker: PhantomData<D>,
}

impl<D> ZCurve<D> {
    pub fn new(num_partitions: usize, order: u32) -> Self {
        Self {
            num_partitions,
            order,
            _marker: PhantomData::<D>,
        }
    }
}

impl<D> Partitioner<D> for ZCurve<D>
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<D>, f64> {
        let points = points.to_points_nd();
        let ids =
            crate::algorithms::z_curve::z_curve_partition(points, self.num_partitions, self.order);
        Partition::from_ids(points, weights, ids)
    }
}

/// # Hilbert space-filling curve algorithm
///
/// An implementation of the Hilbert curve based on
/// "Encoding and Decoding the Hilbert Order" by XIAN LIU and GÜNTHER SCHRACK.
///
/// This algorithm uses space hashing to reorder points alongside the Hilbert curve ov a giver order.
/// See [wikipedia](https://en.wikipedia.org/wiki/Hilbert_curve) for more details.
///
/// # Example
///
/// ```rust
/// use coupe::Point2D;
/// use coupe::Partitioner;
///
/// let points = vec![
///     Point2D::new(0., 0.),
///     Point2D::new(1., 1.),
///     Point2D::new(0., 10.),
///     Point2D::new(1., 9.),
///     Point2D::new(9., 1.),
///     Point2D::new(10., 0.),
///     Point2D::new(10., 10.),
///     Point2D::new(9., 9.),
/// ];
///
/// let weights = vec![1.; 8];
///
/// // generate a partition of 4 parts
/// let hilbert = coupe::HilbertCurve {
///     num_partitions: 4,
///     order: 5,
/// };
///
/// let partition = hilbert.partition(&points, &weights);
/// let ids = partition.ids();
///
/// assert_eq!(ids[0], ids[1]);
/// assert_eq!(ids[2], ids[3]);
/// assert_eq!(ids[4], ids[5]);
/// assert_eq!(ids[6], ids[7]);
/// ```
pub struct HilbertCurve<D> {
    pub num_partitions: usize,
    pub order: u32,
    _marker: PhantomData<D>,
}

impl<D> HilbertCurve<D> {
    pub fn new(num_partitions: usize, order: u32) -> Self {
        Self {
            num_partitions,
            order,
            _marker: PhantomData::<D>,
        }
    }
}

use nalgebra::base::U2;

// hilbert curve is only implemented in 2d for now
impl Partitioner<U2> for HilbertCurve<U2> {
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, U2>,
        _weights: &'a [f64],
    ) -> Partition<'a, PointND<U2>, f64> {
        let points = points.to_points_nd();
        let ids = crate::algorithms::hilbert_curve::hilbert_curve_partition(
            points,
            _weights,
            self.num_partitions,
            self.order as usize,
        );

        Partition::from_ids(points, _weights, ids)
    }
}

/// K-means algorithm
///
/// An implementation of the balanced k-means algorithm inspired from
/// "Balanced k-means for Parallel Geometric Partitioning" by Moritz von Looz,
/// Charilaos Tzovas and Henning Meyerhenke (2018, University of Cologne).
///
/// From an initial partition, the K-means algorithm will generate points clusters that will,
/// at each iteration, exchage points with other clusters that are "closer", and move by recomputing the clusters position (defined as
/// the centroid of the points assigned to the cluster). Eventually the clusters will stop moving, yielding a new partition.
///
/// # Example
///
/// ```rust
/// use coupe::Point2D;
/// use coupe::PartitionImprover;
/// use coupe::ProcessUniqueId;
/// use coupe::partition::Partition;
///
/// // create ids for initial partition
/// let p1 = ProcessUniqueId::new();
/// let p2 = ProcessUniqueId::new();
/// let p3 = ProcessUniqueId::new();
///
/// let points = vec![
///     Point2D::new(0., 0.),
///     Point2D::new(1., 0.),
///     Point2D::new(2., 0.),
///     Point2D::new(0., 5.),
///     Point2D::new(1., 5.),
///     Point2D::new(2., 5.),
///     Point2D::new(0., 10.),
///     Point2D::new(1., 10.),
///     Point2D::new(2., 10.),
/// ];
///
/// let weights = vec![1.; 9];
///
/// // create an unbalanced partition:
/// //  - p1: total weight = 1
/// //  - p2: total weight = 7
/// //  - p3: total weight = 1
/// let ids = vec![p1, p2, p2, p2, p2, p2, p2, p2, p3];
/// let partition = Partition::from_ids(&points, &weights, ids);
///
/// let k_means = coupe::KMeans {
///     num_partitions: 3,
///     delta_threshold: 0.,
///     ..Default::default()
/// };
///
/// let partition = k_means.improve_partition(partition);
/// let ids = partition.ids();
///
/// assert_eq!(ids[0], ids[1]);
/// assert_eq!(ids[0], ids[2]);
///
/// assert_eq!(ids[3], ids[4]);
/// assert_eq!(ids[3], ids[5]);
///
/// assert_eq!(ids[6], ids[7]);
/// assert_eq!(ids[6], ids[8]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KMeans {
    pub num_partitions: usize,
    pub imbalance_tol: f64,
    pub delta_threshold: f64,
    pub max_iter: usize,
    pub max_balance_iter: usize,
    pub erode: bool,
    pub hilbert: bool,
    pub mbr_early_break: bool,
}

impl Default for KMeans {
    fn default() -> Self {
        Self {
            num_partitions: 7,
            imbalance_tol: 5.,
            delta_threshold: 0.01,
            max_iter: 500,
            max_balance_iter: 20, // for now, `max_balance_iter > 1` yields poor convergence time
            erode: false,         // for now, `erode` yields` enabled yields wrong results
            hilbert: true,
            mbr_early_break: false, // for now, `mbr_early_break` enabled yields wrong results
        }
    }
}

impl<D> PartitionImprover<D> for KMeans
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    fn improve_partition<'a>(
        &self,
        partition: Partition<'a, PointND<D>, f64>,
    ) -> Partition<'a, PointND<D>, f64> {
        let settings = crate::algorithms::k_means::BalancedKmeansSettings {
            num_partitions: self.num_partitions,
            imbalance_tol: self.imbalance_tol,
            delta_threshold: self.delta_threshold,
            max_iter: self.max_iter,
            max_balance_iter: self.max_balance_iter,
            erode: self.erode,
            hilbert: self.hilbert,
            mbr_early_break: self.mbr_early_break,
        };
        let (points, weights, mut ids) = partition.into_raw();
        crate::algorithms::k_means::balanced_k_means_with_initial_partition(
            points, weights, settings, &mut ids,
        );
        Partition::from_ids(points, weights, ids)
    }
}

/// # Represents the composition algorithm.
///
/// This structure is created by calling the [`compose`](trait.Compose.html#tymethod.compose)
/// of the [`Compose`](trait.Compose.html) trait.
pub struct Composition<T, U, D> {
    first: T,
    second: U,
    _marker: PhantomData<D>,
}

impl<T, U, D> Composition<T, U, D> {
    pub fn new(first: T, second: U) -> Self {
        Self {
            first,
            second,
            _marker: PhantomData::<D>,
        }
    }
}

impl<D, T, U> Partitioner<D> for Composition<T, U, D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    T: Partitioner<D>,
    U: PartitionImprover<D>,
{
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<D>, f64> {
        let points = points.to_points_nd();
        let partition = self.first.partition(points, weights);
        self.second.improve_partition(partition)
    }
}

impl<D, T, U> PartitionImprover<D> for Composition<T, U, D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    T: PartitionImprover<D>,
    U: PartitionImprover<D>,
{
    fn improve_partition<'a>(
        &self,
        partition: Partition<'a, PointND<D>, f64>,
    ) -> Partition<'a, PointND<D>, f64> {
        self.second
            .improve_partition(self.first.improve_partition(partition))
    }
}

/// # Compose two algorithms.
///
/// This trait enables algorithms to be composed together. Doing so allows for instance to improve the partition
/// generated by a [`Partitioner`](trait.Partitioner.html) with a [`PartitionImprover`](trait.PartitionImprover.html).
///
/// The resulting composition implements either [`Partitioner`](trait.Partitioner.html) or [`PartitionImprover`](trait.PartitionImprover.html)
/// based on the input algorithms.
///
/// # Example
/// ```rust
/// use coupe::{Compose, Partitioner, PartitionImprover};
/// use coupe::dimension::U3;
///
/// let multi_jagged_then_k_means = coupe::MultiJagged {
///     num_partitions: 7,
///     max_iter: 3,
/// }.compose::<U3>( // dimension is required
///     coupe::KMeans {
///         num_partitions: 7,
///         ..Default::default()
///     }
/// );
///
/// ```
pub trait Compose<T, D> {
    type Composed;
    fn compose(self, other: T) -> Self::Composed
    where
        D: DimName,
        DefaultAllocator: Allocator<f64, D>;
}

impl<T, U, D> Compose<T, D> for U {
    type Composed = Composition<Self, T, D>;
    fn compose(self, other: T) -> Self::Composed {
        Composition::new(self, other)
    }
}
