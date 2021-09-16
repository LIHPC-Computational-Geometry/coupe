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
mod num;
pub mod partition;
pub mod topology;

#[cfg(test)]
mod tests;

// API

// SUBMODULES REEXPORT
pub use crate::geometry::Matrix;
pub use crate::geometry::Point2D;
pub use crate::geometry::Point3D;
pub use crate::geometry::PointND;
pub use crate::num::Sqrt;
pub use crate::num::Two;
pub use snowflake::ProcessUniqueId;

use ndarray::ArrayView1;
use ndarray::ArrayView2;
use sprs::CsMatView;

use std::marker::PhantomData;

use crate::partition::*;

// Trait that allows conversions from/to different kinds of
// points views representation as partitioner inputs
// e.g. &[f64], &[PointND<f64, D>], slice from ndarray, ...
pub trait PointsView<'a, const D: usize> {
    fn to_points_nd(self) -> &'a [PointND<f64, D>];
}

impl<'a, const D: usize> PointsView<'a, D> for &'a [f64] {
    fn to_points_nd(self) -> &'a [PointND<f64, D>] {
        if self.len() % D != 0 {
            panic!("error: tried to convert a &[f64] to a &[PointND<f64, D>] with D = {}, but input slice has len {}", D, self.len());
        }
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, self.len() / D) }
    }
}

impl<'a, const D: usize> PointsView<'a, D> for ArrayView1<'a, f64> {
    fn to_points_nd(self) -> &'a [PointND<f64, D>] {
        let slice = self.to_slice().expect(
            "Cannot convert an ArrayView1 with dicontiguous storage repr to a slice. Try cloning the data into a contiguous array first"
        );
        slice.to_points_nd()
    }
}

impl<'a, const D: usize> PointsView<'a, D> for ArrayView2<'a, f64> {
    fn to_points_nd(self) -> &'a [PointND<f64, D>] {
        let slice = self.to_slice().expect(
            "Cannot convert an ArrayView2 with dicontiguous storage repr to a slice. Try cloning the data into a contiguous array first"
        );
        slice.to_points_nd()
    }
}

impl<'a, const D: usize> PointsView<'a, D> for &'a [PointND<f64, D>] {
    fn to_points_nd(self) -> &'a [PointND<f64, D>] {
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
pub trait Partitioner<const D: usize> {
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<f64, D>, f64>;
}

/// A geometric algorithm to improve a partition.
///
/// Algorithms that implement [`PartitionImprover`](trait.PartitionImprover.html) operate on a set of geometric
/// points and associated weights to modify and improve an existing partition (typically generated by a [`Partitioner`](trait.Partitioner.html)).
///
/// See the [implementors](trait.PartitionImprover.html#implementors) for more information about the currently available algorithms.
pub trait PartitionImprover<const D: usize> {
    fn improve_partition<'a>(
        &self,
        partition: Partition<'a, PointND<f64, D>, f64>,
    ) -> Partition<'a, PointND<f64, D>, f64>;
}

pub trait TopologicPartitioner<const D: usize> {
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
        adjacency: CsMatView<f64>,
    ) -> Partition<'a, PointND<f64, D>, f64>;
}

pub trait TopologicPartitionImprover<const D: usize> {
    fn improve_partition<'a>(
        &self,
        partition: Partition<'a, PointND<f64, D>, f64>,
        adjacency: CsMatView<f64>,
    ) -> Partition<'a, PointND<f64, D>, f64>;
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
/// let rcb = coupe::Rcb::new(2);
/// let partition = rcb.partition(points.as_slice(), &weights);
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
pub struct Rcb<const D: usize> {
    pub num_iter: usize,
    _marker: PhantomData<[u8; D]>,
}

impl<const D: usize> Rcb<D> {
    pub fn new(num_iter: usize) -> Self {
        Self {
            num_iter,
            _marker: PhantomData,
        }
    }
}

impl<const D: usize> Partitioner<D> for Rcb<D> {
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<f64, D>, f64> {
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
/// let rib = coupe::Rib::new(1);
/// let partition = rib.partition(points.as_slice(), &weights);
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
pub struct Rib<const D: usize> {
    /// The number of iterations of the algorithm. This will yield a partition of `2^num_iter` parts.
    pub num_iter: usize,
    _marker: PhantomData<[u8; D]>,
}

impl<const D: usize> Rib<D> {
    pub fn new(num_iter: usize) -> Self {
        Self {
            num_iter,
            _marker: PhantomData,
        }
    }
}

impl<const D: usize> Partitioner<D> for Rib<D> {
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<f64, D>, f64> {
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
/// let num_partitions = 9;
/// let max_iter = 4;
///
/// // generate a partition of 4 parts
/// let multi_jagged = coupe::MultiJagged::new(num_partitions, max_iter);
///
/// let partition = multi_jagged.partition(points.as_slice(), &weights);
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
pub struct MultiJagged<const D: usize> {
    pub num_partitions: usize,
    pub max_iter: usize,
    _marker: PhantomData<[u8; D]>,
}

impl<const D: usize> MultiJagged<D> {
    pub fn new(num_partitions: usize, max_iter: usize) -> Self {
        Self {
            num_partitions,
            max_iter,
            _marker: PhantomData,
        }
    }
}

impl<const D: usize> Partitioner<D> for MultiJagged<D> {
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<f64, D>, f64> {
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
/// let num_partitions = 4;
/// let order = 5;
///
/// // generate a partition of 4 parts
/// let z_curve = coupe::ZCurve::new(num_partitions, order);
///
/// let partition = z_curve.partition(points.as_slice(), &weights);
/// let ids = partition.ids();
///
/// assert_eq!(ids[0], ids[1]);
/// assert_eq!(ids[2], ids[3]);
/// assert_eq!(ids[4], ids[5]);
/// assert_eq!(ids[6], ids[7]);
/// ```  
pub struct ZCurve<const D: usize> {
    pub num_partitions: usize,
    pub order: u32,
    _marker: PhantomData<[u8; D]>,
}

impl<const D: usize> ZCurve<D> {
    pub fn new(num_partitions: usize, order: u32) -> Self {
        Self {
            num_partitions,
            order,
            _marker: PhantomData,
        }
    }
}

impl<const D: usize> Partitioner<D> for ZCurve<D> {
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<f64, D>, f64> {
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
/// let num_partitions = 4;
/// let order = 5;
///
/// // generate a partition of 4 parts
/// let hilbert = coupe::HilbertCurve::new(num_partitions, order);
///
/// let partition = hilbert.partition(points.as_slice(), &weights);
/// let ids = partition.ids();
///
/// assert_eq!(ids[0], ids[1]);
/// assert_eq!(ids[2], ids[3]);
/// assert_eq!(ids[4], ids[5]);
/// assert_eq!(ids[6], ids[7]);
/// ```
pub struct HilbertCurve<const D: usize> {
    pub num_partitions: usize,
    pub order: u32,
    _marker: PhantomData<[u8; D]>,
}

impl<const D: usize> HilbertCurve<D> {
    pub fn new(num_partitions: usize, order: u32) -> Self {
        Self {
            num_partitions,
            order,
            _marker: PhantomData,
        }
    }
}

// hilbert curve is only implemented in 2d for now
impl Partitioner<2> for HilbertCurve<2> {
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, 2>,
        _weights: &'a [f64],
    ) -> Partition<'a, PointND<f64, 2>, f64> {
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
/// let mut k_means = coupe::KMeans::default();
/// k_means.num_partitions = 3;
/// k_means.delta_threshold = 0.;
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
pub struct KMeans<const D: usize> {
    pub num_partitions: usize,
    pub imbalance_tol: f64,
    pub delta_threshold: f64,
    pub max_iter: usize,
    pub max_balance_iter: usize,
    pub erode: bool,
    pub hilbert: bool,
    pub mbr_early_break: bool,
    _marker: PhantomData<[u8; D]>,
}

// KMeans builder pattern
// to reduce construction boilerplate
// e.g.
// ```rust
// let k_means = KMeansBuilder::default()
//    .imbalance_tol(5.)
//    .max_balance_iter(12)
//    .build();
// ```
#[derive(Debug, Clone, Copy)]
pub struct KMeansBuilder<const D: usize> {
    inner: KMeans<D>,
}

impl<const D: usize> Default for KMeansBuilder<D> {
    fn default() -> Self {
        Self {
            inner: KMeans::default(),
        }
    }
}

impl<const D: usize> KMeansBuilder<D> {
    pub fn build(self) -> KMeans<D> {
        self.inner
    }

    pub fn num_partitions(mut self, num_partitions: usize) -> Self {
        self.inner.num_partitions = num_partitions;
        self
    }

    pub fn imbalance_tol(mut self, imbalance_tol: f64) -> Self {
        self.inner.imbalance_tol = imbalance_tol;
        self
    }

    pub fn delta_threshold(mut self, delta_threshold: f64) -> Self {
        self.inner.delta_threshold = delta_threshold;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.inner.max_iter = max_iter;
        self
    }

    pub fn max_balance_iter(mut self, max_balance_iter: usize) -> Self {
        self.inner.max_balance_iter = max_balance_iter;
        self
    }

    pub fn erode(mut self, erode: bool) -> Self {
        self.inner.erode = erode;
        self
    }

    pub fn hilbert(mut self, hilbert: bool) -> Self {
        self.inner.hilbert = hilbert;
        self
    }

    pub fn mbr_early_break(mut self, mbr_early_break: bool) -> Self {
        self.inner.mbr_early_break = mbr_early_break;
        self
    }
}

impl<const D: usize> KMeans<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_partitions: usize,
        imbalance_tol: f64,
        delta_threshold: f64,
        max_iter: usize,
        max_balance_iter: usize,
        erode: bool,
        hilbert: bool,
        mbr_early_break: bool,
    ) -> Self {
        Self {
            num_partitions,
            imbalance_tol,
            delta_threshold,
            max_iter,
            max_balance_iter,
            erode,
            hilbert,
            mbr_early_break,
            _marker: PhantomData,
        }
    }
}

impl<const D: usize> Default for KMeans<D> {
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
            _marker: PhantomData,
        }
    }
}

impl<const D: usize> PartitionImprover<D> for KMeans<D> {
    fn improve_partition<'a>(
        &self,
        partition: Partition<'a, PointND<f64, D>, f64>,
    ) -> Partition<'a, PointND<f64, D>, f64> {
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

/// KernighanLin algorithm
///
/// An implementation of the Kernighan Lin topologic algorithm
/// for graph partitioning. The current implementation currently only handles
/// partitioning a graph into two parts, as described in the original algorithm in
/// "An efficient heuristic procedure for partitioning graphs" by W. Kernighan and S. Lin.
///
/// The algorithms repeats an iterative pass during which several pairs of nodes have
/// their part assignment swapped in order to reduce the cutsize of the partition.
/// If all the nodes are equally weighted, the algorithm preserves the partition balance.
///
/// # Example
///
/// ```rust
/// use coupe::{Point2D, ProcessUniqueId};
/// use coupe::TopologicPartitionImprover;
/// use coupe::partition::Partition;
/// use sprs::CsMat;
///
/// //    swap
/// // 0  1  0  1
/// // +--+--+--+
/// // |  |  |  |
/// // +--+--+--+
/// // 0  0  1  1
/// let points = vec![
///      Point2D::new(0., 0.),
///      Point2D::new(1., 0.),
///      Point2D::new(2., 0.),
///      Point2D::new(3., 0.),
///      Point2D::new(0., 1.),
///      Point2D::new(1., 1.),
///      Point2D::new(2., 1.),
///      Point2D::new(3., 1.),
///  ];
///  let id0 = ProcessUniqueId::new();
///  let id1 = ProcessUniqueId::new();
///
///  let ids = vec![id0, id0, id1, id1, id0, id1, id0, id1];
///  let weights = vec![1.; 8];
///
///  let mut partition = Partition::from_ids(&points, &weights, ids);
///
///  let mut adjacency = CsMat::empty(sprs::CSR, 8);
///  adjacency.reserve_outer_dim(8);
///  eprintln!("shape: {:?}", adjacency.shape());
///  adjacency.insert(0, 1, 1.);
///  adjacency.insert(1, 2, 1.);
///  adjacency.insert(2, 3, 1.);
///  adjacency.insert(4, 5, 1.);
///  adjacency.insert(5, 6, 1.);
///  adjacency.insert(6, 7, 1.);
///  adjacency.insert(0, 4, 1.);
///  adjacency.insert(1, 5, 1.);
///  adjacency.insert(2, 6, 1.);
///  adjacency.insert(3, 7, 1.);
///  
///  // symmetry
///  adjacency.insert(1, 0, 1.);
///  adjacency.insert(2, 1, 1.);
///  adjacency.insert(3, 2, 1.);
///  adjacency.insert(5, 4, 1.);
///  adjacency.insert(6, 5, 1.);
///  adjacency.insert(7, 6, 1.);
///  adjacency.insert(4, 0, 1.);
///  adjacency.insert(5, 1, 1.);
///  adjacency.insert(6, 2, 1.);
///  adjacency.insert(7, 3, 1.);
///
/// // 1 iteration
/// let algo = coupe::KernighanLin::new(1, 1, None, 1);
///
/// let partition = algo.improve_partition(partition, adjacency.view());
///
/// let new_ids = partition.into_ids();
/// assert_eq!(new_ids[5], id0);
/// assert_eq!(new_ids[6], id1);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KernighanLin {
    max_passes: Option<usize>,
    max_flips_per_pass: Option<usize>,
    max_imbalance_per_flip: Option<f64>,
    max_bad_move_in_a_row: usize,
}

impl KernighanLin {
    pub fn new(
        max_passes: impl Into<Option<usize>>,
        max_flips_per_pass: impl Into<Option<usize>>,
        max_imbalance_per_flip: impl Into<Option<f64>>,
        max_bad_move_in_a_row: usize,
    ) -> Self {
        Self {
            max_passes: max_passes.into(),
            max_flips_per_pass: max_flips_per_pass.into(),
            max_imbalance_per_flip: max_imbalance_per_flip.into(),
            max_bad_move_in_a_row,
        }
    }
}

impl<const D: usize> TopologicPartitionImprover<D> for KernighanLin {
    fn improve_partition<'a>(
        &self,
        mut partition: Partition<'a, PointND<f64, D>, f64>,
        adjacency: CsMatView<f64>,
    ) -> Partition<'a, PointND<f64, D>, f64> {
        crate::algorithms::kernighan_lin::kernighan_lin(
            &mut partition,
            adjacency,
            self.max_passes,
            self.max_flips_per_pass,
            self.max_imbalance_per_flip,
            self.max_bad_move_in_a_row,
        );
        partition
    }
}

/// FiducciaMattheyses
///
/// An implementation of the Fiduccia Mattheyses topologic algorithm
/// for graph partitioning. This implementation is an extension of the
/// original algorithm to handle partitioning into more than two parts.
///
/// This algorithm repeats an iterative pass during which a set of graph nodes are assigned to
/// a new part, reducing the overall cutsize of the partition. As opposed to the
/// Kernighan-Lin algorithm, during each pass iteration, only one node is flipped at a time.
/// The algorithm thus does not preserve partition weights balance and may produce an unbalanced
/// partition.
///
/// Original algorithm from "A Linear-Time Heuristic for Improving Network Partitions"
/// by C.M. Fiduccia and R.M. Mattheyses.
///
/// # Example
///
/// ```rust
/// use coupe::{Point2D, ProcessUniqueId};
/// use coupe::TopologicPartitionImprover;
/// use coupe::partition::Partition;
/// use sprs::CsMat;
///
/// //    swap
/// // 0  1  0  1
/// // +--+--+--+
/// // |  |  |  |
/// // +--+--+--+
/// // 0  0  1  1
/// let points = vec![
///      Point2D::new(0., 0.),
///      Point2D::new(1., 0.),
///      Point2D::new(2., 0.),
///      Point2D::new(3., 0.),
///      Point2D::new(0., 1.),
///      Point2D::new(1., 1.),
///      Point2D::new(2., 1.),
///      Point2D::new(3., 1.),
///  ];
///  let id0 = ProcessUniqueId::new();
///  let id1 = ProcessUniqueId::new();
///
///  let ids = vec![id0, id0, id1, id1, id0, id1, id0, id1];
///  let weights = vec![1.; 8];
///
///  let mut partition = Partition::from_ids(&points, &weights, ids);
///
///  let mut adjacency = CsMat::empty(sprs::CSR, 8);
///  adjacency.reserve_outer_dim(8);
///  eprintln!("shape: {:?}", adjacency.shape());
///  adjacency.insert(0, 1, 1.);
///  adjacency.insert(1, 2, 1.);
///  adjacency.insert(2, 3, 1.);
///  adjacency.insert(4, 5, 1.);
///  adjacency.insert(5, 6, 1.);
///  adjacency.insert(6, 7, 1.);
///  adjacency.insert(0, 4, 1.);
///  adjacency.insert(1, 5, 1.);
///  adjacency.insert(2, 6, 1.);
///  adjacency.insert(3, 7, 1.);
///  
///  // symmetry
///  adjacency.insert(1, 0, 1.);
///  adjacency.insert(2, 1, 1.);
///  adjacency.insert(3, 2, 1.);
///  adjacency.insert(5, 4, 1.);
///  adjacency.insert(6, 5, 1.);
///  adjacency.insert(7, 6, 1.);
///  adjacency.insert(4, 0, 1.);
///  adjacency.insert(5, 1, 1.);
///  adjacency.insert(6, 2, 1.);
///  adjacency.insert(7, 3, 1.);
///
/// // 1 iteration
/// let algo = coupe::FiducciaMattheyses::new(None, None, None, 1);
///
/// let partition = algo.improve_partition(partition, adjacency.view());
///
/// let new_ids = partition.into_ids();
/// assert_eq!(new_ids[5], id0);
/// assert_eq!(new_ids[6], id1);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FiducciaMattheyses {
    max_passes: Option<usize>,
    max_flips_per_pass: Option<usize>,
    max_imbalance_per_flip: Option<f64>,
    max_bad_move_in_a_row: usize,
}

impl FiducciaMattheyses {
    pub fn new(
        max_passes: impl Into<Option<usize>>,
        max_flips_per_pass: impl Into<Option<usize>>,
        max_imbalance_per_flip: impl Into<Option<f64>>,
        max_bad_move_in_a_row: usize,
    ) -> Self {
        Self {
            max_passes: max_passes.into(),
            max_flips_per_pass: max_flips_per_pass.into(),
            max_imbalance_per_flip: max_imbalance_per_flip.into(),
            max_bad_move_in_a_row,
        }
    }
}

impl<const D: usize> TopologicPartitionImprover<D> for FiducciaMattheyses {
    fn improve_partition<'a>(
        &self,
        mut partition: Partition<'a, PointND<f64, D>, f64>,
        adjacency: CsMatView<f64>,
    ) -> Partition<'a, PointND<f64, D>, f64> {
        crate::algorithms::fiduccia_mattheyses::fiduccia_mattheyses(
            &mut partition,
            adjacency,
            self.max_passes,
            self.max_flips_per_pass,
            self.max_imbalance_per_flip,
            self.max_bad_move_in_a_row,
        );
        partition
    }
}

/// Graph Growth algorithm
///
/// A topologic algorithm that generates a partition from a topologic mesh.
/// Given a number k of parts, the algorithm selects k nodes randomly and assigns them to a different part.
/// Then, at each iteration, each part is expanded to neighbor nodes that are not yet assigned to a part
///
/// # Example
///
/// ```rust
/// use coupe::{Point2D, ProcessUniqueId};
/// use coupe::TopologicPartitioner;
/// use coupe::partition::Partition;
/// use sprs::CsMat;
///
/// // +--+--+--+
/// // |  |  |  |
/// // +--+--+--+
///
/// let points = vec![
///      Point2D::new(0., 0.),
///      Point2D::new(1., 0.),
///      Point2D::new(2., 0.),
///      Point2D::new(3., 0.),
///      Point2D::new(0., 1.),
///      Point2D::new(1., 1.),
///      Point2D::new(2., 1.),
///      Point2D::new(3., 1.),
///  ];
///
///  let weights = vec![1.; 8];
///
///  let mut adjacency = CsMat::empty(sprs::CSR, 8);
///  adjacency.reserve_outer_dim(8);
///  eprintln!("shape: {:?}", adjacency.shape());
///  adjacency.insert(0, 1, 1.);
///  adjacency.insert(1, 2, 1.);
///  adjacency.insert(2, 3, 1.);
///  adjacency.insert(4, 5, 1.);
///  adjacency.insert(5, 6, 1.);
///  adjacency.insert(6, 7, 1.);
///  adjacency.insert(0, 4, 1.);
///  adjacency.insert(1, 5, 1.);
///  adjacency.insert(2, 6, 1.);
///  adjacency.insert(3, 7, 1.);
///  
///  // symmetry
///  adjacency.insert(1, 0, 1.);
///  adjacency.insert(2, 1, 1.);
///  adjacency.insert(3, 2, 1.);
///  adjacency.insert(5, 4, 1.);
///  adjacency.insert(6, 5, 1.);
///  adjacency.insert(7, 6, 1.);
///  adjacency.insert(4, 0, 1.);
///  adjacency.insert(5, 1, 1.);
///  adjacency.insert(6, 2, 1.);
///  adjacency.insert(7, 3, 1.);
///
/// let gg = coupe::GraphGrowth::new(2);
///
/// let _partition = gg.partition(points.as_slice(), &weights, adjacency.view());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GraphGrowth {
    num_partitions: usize,
}

impl GraphGrowth {
    pub fn new(num_partitions: usize) -> Self {
        Self { num_partitions }
    }
}

impl<const D: usize> TopologicPartitioner<D> for GraphGrowth {
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
        adjacency: CsMatView<f64>,
    ) -> Partition<'a, PointND<f64, D>, f64> {
        let ids = crate::algorithms::graph_growth::graph_growth(
            weights,
            adjacency.view(),
            self.num_partitions,
        );
        Partition::from_ids(points.to_points_nd(), weights, ids)
    }
}

/// # Represents the composition algorithm.
///
/// This structure is created by calling the [`compose`](trait.Compose.html#tymethod.compose)
/// of the [`Compose`](trait.Compose.html) trait.
pub struct Composition<T, U> {
    first: T,
    second: U,
}

impl<T, U> Composition<T, U> {
    pub fn new(first: T, second: U) -> Self {
        Self { first, second }
    }
}

impl<T, U, const D: usize> Partitioner<D> for Composition<T, U>
where
    T: Partitioner<D>,
    U: PartitionImprover<D>,
{
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
    ) -> Partition<'a, PointND<f64, D>, f64> {
        let points = points.to_points_nd();
        let partition = self.first.partition(points, weights);
        self.second.improve_partition(partition)
    }
}

impl<T, U, const D: usize> PartitionImprover<D> for Composition<T, U>
where
    T: PartitionImprover<D>,
    U: PartitionImprover<D>,
{
    fn improve_partition<'a>(
        &self,
        partition: Partition<'a, PointND<f64, D>, f64>,
    ) -> Partition<'a, PointND<f64, D>, f64> {
        self.second
            .improve_partition(self.first.improve_partition(partition))
    }
}

impl<T, U, const D: usize> TopologicPartitioner<D> for Composition<T, U>
where
    T: Partitioner<D>,
    U: TopologicPartitionImprover<D>,
{
    fn partition<'a>(
        &self,
        points: impl PointsView<'a, D>,
        weights: &'a [f64],
        adjacency: CsMatView<f64>,
    ) -> Partition<'a, PointND<f64, D>, f64> {
        let partition = self.first.partition(points, weights);
        self.second.improve_partition(partition, adjacency)
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
/// let num_partitions = 7;
/// let max_iter = 3;
///
/// let mut k_means = coupe::KMeans::<U3>::default();
/// k_means.num_partitions = 7;
/// let multi_jagged_then_k_means = coupe::MultiJagged::<U3>::new(
///     num_partitions,
///     max_iter
/// ).compose(k_means);
///
/// ```
pub trait Compose<T> {
    type Composed;
    fn compose(self, other: T) -> Self::Composed;
}

impl<T, U> Compose<T> for U {
    type Composed = Composition<Self, T>;
    fn compose(self, other: T) -> Self::Composed {
        Composition::new(self, other)
    }
}
