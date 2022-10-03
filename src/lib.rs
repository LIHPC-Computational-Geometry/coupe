//! A mesh partitioning library that implements multithreaded, composable geometric algorithms.
//!
//! # Crate Layout
//!
//! Coupe exposes a [`Partition`] trait, which is in turn implemented by
//! algorithms.  See its documentation for more details.  The trait is generic around its input, which means algorithms
//! can partition different type of collections (e.g. 2D and 3D meshes).
//!
//! # Available algorithms
//!
//! ## Partitioner algorithms
//!
//! - Space filling curves:
//!   + [Z-curve][ZCurve]
//!   + [Hilbert curve][HilbertCurve]
//! - [Recursive Coordinate Bisection][Rcb]
//! - [Recursive Inertial Bisection][Rib]
//! - [Multi jagged][MultiJagged]
//! - Number partitioning:
//!   + [Greedy][Greedy]
//!   + [Karmarkar-Karp][KarmarkarKarp] and its [complete][CompleteKarmarkarKarp] version
//!
//! ## Partition improving algorithms
//!
//! - [K-means][KMeans]
//! - Number partitioning:
//!   + [VN-Best][VnBest]
//!   + [VN-First][VnFirst]
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
pub use crate::geometry::BoundingBox;
pub use crate::geometry::{Point2D, Point3D, PointND};
pub use crate::nextafter::nextafter;
pub use crate::real::Real;

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
