//! This package that implements multithreaded, composable partitioning algorithms.
//!
//! It contains methods to solve:
//! - [geometric partitioning](#geometric-partitioning)
//! - [number partitioning](#number-partitioning)
//! - [topological partitioning](#topological-partitioning)
//!
//! Coupe exposes a [`Partition`] trait, which is in turn implemented by
//! algorithms.  See its documentation for more details.  The trait is generic around its input, which means algorithms
//! can partition different type of collections (e.g. 2D and 3D meshes).
//!
//! # Library Use
//!
//! ```rust
//! extern crate coupe;
//! use coupe::Partition as _;
//! use coupe::Point2D;
//!
//! // define coordinates, weights and graph
//! # let coordinates: [Point2D; 9] = [
//! #        Point2D::new(0.0, 0.0),
//! #        Point2D::new(0.0, 1.0),
//! #        Point2D::new(0.0, 2.0),
//! #        Point2D::new(1.0, 0.0),
//! #        Point2D::new(1.0, 1.0),
//! #        Point2D::new(1.0, 2.0),
//! #        Point2D::new(2.0, 0.0),
//! #        Point2D::new(2.0, 1.0),
//! #        Point2D::new(2.0, 2.0),
//! # ];
//! # let weights: [f64; 9] = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0];
//! # let graph: sprs::CsMat<i64> = {
//! # let mut g = sprs::CsMat::empty(sprs::CSR, 9);
//! # g.insert(0, 1, 1);
//! # g.insert(0, 3, 1);
//! # g.insert(1, 0, 1);
//! # g.insert(1, 2, 1);
//! # g.insert(1, 4, 1);
//! # g.insert(2, 1, 1);
//! # g.insert(2, 5, 1);
//! # g.insert(3, 0, 1);
//! # g.insert(3, 4, 1);
//! # g.insert(3, 6, 1);
//! # g.insert(4, 1, 1);
//! # g.insert(4, 3, 1);
//! # g.insert(4, 5, 1);
//! # g.insert(4, 7, 1);
//! # g.insert(5, 2, 1);
//! # g.insert(5, 4, 1);
//! # g.insert(5, 8, 1);
//! # g.insert(6, 3, 1);
//! # g.insert(6, 7, 1);
//! # g.insert(7, 4, 1);
//! # g.insert(7, 6, 1);
//! # g.insert(7, 8, 1);
//! # g.insert(8, 5, 1);
//! # g.insert(8, 7, 1);
//! # g
//! # };
//! ```
//!
//! ## Geometric Partitioning
//!
//! - Space filling curves:
//!   + [Z-curve][ZCurve]
//!   + [Hilbert curve][HilbertCurve]
//! - Recursive bisections:
//!   + [Recursive Coordinate Bisection][Rcb]
//!   + [Recursive Inertial Bisection][Rib]
//!   + [Multi jagged][MultiJagged]
//! - Local optimisation:
//!   + [K-means][KMeans]
//!
//! ## Number Partitioning
//! - Direct:
//!   + [Greedy][Greedy]
//!   + [Karmarkar-Karp][KarmarkarKarp] and its [complete][CompleteKarmarkarKarp] version
//! - Local optimisation:
//!   + [VN-Best][VnBest]
//!   + [VN-First][VnFirst]
//!
//! ## Topological Partitioning
//!
//! - [Fiduccia-Mattheyses][FiducciaMattheyses]
//! - [Kernighan-Lin][KernighanLin]

#![warn(
    missing_copy_implementations,
    missing_debug_implementations,
    rust_2018_idioms
)]

mod algorithms;
mod defer;
mod geometry;
pub mod imbalance;
mod nextafter;
mod real;
pub mod topology;
mod work_share;

pub use crate::algorithms::*;
pub use crate::geometry::{BoundingBox, Point2D, Point3D, PointND};
pub use crate::real::Real;

// Internal use
use crate::nextafter::nextafter;
use crate::work_share::work_share;

pub use nalgebra;
pub use num_traits;
pub use rayon;
pub use sprs;

use std::cmp::Ordering;

/// The `Partition` trait allows for partitioning data.
///
/// Partitioning algorithms implement this trait.
///
/// The generic argument `M` defines the input of the algorithms (e.g. an
/// adjacency matrix or a 2D set of points).
///
/// The input partition must be of the correct size and its contents may or may
/// not be used by the algorithms.
pub trait Partition<M> {
    /// Diagnostic data returned for a specific run of the algorithm.
    type Metadata;

    /// Error details, should the algorithm fail to run.
    type Error;

    /// Partition the given data and output the part ID of each element in
    /// `part_ids`.
    ///
    /// Part IDs must be contiguous and start from zero, meaning the number of
    /// parts is one plus the maximum of `part_ids`.  If a lower ID does not
    /// appear in the array, the part is assumed to be empty.
    fn partition(&mut self, part_ids: &mut [usize], data: M)
        -> Result<Self::Metadata, Self::Error>;
}

fn partial_cmp<W>(a: &W, b: &W) -> Ordering
where
    W: PartialOrd,
{
    if a < b {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}
