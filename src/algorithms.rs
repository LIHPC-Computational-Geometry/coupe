use std::fmt;

mod ckk;
mod fiduccia_mattheyses;
mod graph_growth;
mod greedy;
mod hilbert_curve;
mod k_means;
mod kernighan_lin;
mod kk;
mod multi_jagged;
mod recursive_bisection;
mod vn;
mod z_curve;

pub use ckk::CkkWeight;
pub use ckk::CompleteKarmarkarKarp;
pub use fiduccia_mattheyses::FiducciaMattheyses;
pub use fiduccia_mattheyses::FmWeight;
pub use fiduccia_mattheyses::Metadata as FmMetadata;
pub use graph_growth::GraphGrowth;
pub use greedy::Greedy;
pub use greedy::GreedyWeight;
pub use hilbert_curve::Error as HilbertCurveError;
pub use hilbert_curve::HilbertCurve;
pub use k_means::KMeans;
pub use kernighan_lin::KernighanLin;
pub use kk::KarmarkarKarp;
pub use kk::KkWeight;
pub use multi_jagged::MultiJagged;
pub use recursive_bisection::Rcb;
pub use recursive_bisection::RcbWeight;
pub use recursive_bisection::Rib;
use std::collections::TryReserveError;
pub use vn::VnBest;
pub use vn::VnBestWeight;
pub use vn::VnFirst;
pub use vn::VnFirstWeight;
pub use z_curve::ZCurve;

/// Common errors thrown by algorithms.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Error {
    /// An allocation failed, out of memory error.
    Alloc,

    /// No partition that matches the given criteria could been found.
    NotFound,

    /// Input sets don't have matching lengths.
    InputLenMismatch { expected: usize, actual: usize },

    /// Input contains negative values and such values are not supported.
    NegativeValues,

    /// When a partition improving algorithm is given more than 2 parts.
    BiPartitioningOnly,

    /// Conversion between types failed.
    Conversion {
        src_type: &'static str,
        dst_type: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc => write!(f, "out of memory"),
            Self::NotFound => write!(f, "no partition found"),
            Self::InputLenMismatch { expected, actual } => write!(
                f,
                "input sets don't have the same length (expected {expected} items, got {actual})",
            ),
            Self::NegativeValues => write!(f, "input contains negative values"),
            Self::BiPartitioningOnly => write!(f, "expected no more than two parts"),
            Self::Conversion { src_type, dst_type } => {
                write!(f, "failed conversion from {src_type} to {dst_type}")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<TryReserveError> for Error {
    fn from(_: TryReserveError) -> Self {
        Self::Alloc
    }
}

fn try_from_f64<T>(f: f64) -> Result<T, Error>
where
    T: num::FromPrimitive,
{
    T::from_f64(f).ok_or_else(|| Error::Conversion {
        src_type: std::any::type_name::<f64>(),
        dst_type: std::any::type_name::<T>(),
    })
}

fn try_to_f64<T>(t: &T) -> Result<f64, Error>
where
    T: num::ToPrimitive,
{
    T::to_f64(t).ok_or_else(|| Error::Conversion {
        src_type: std::any::type_name::<T>(),
        dst_type: std::any::type_name::<f64>(),
    })
}

/// Map elements to parts randomly.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), std::convert::Infallible> {
/// use coupe::Partition as _;
/// use rand;
///
/// let mut partition = [0; 12];
///
/// coupe::Random { rng: rand::thread_rng(), part_count: 3 }
///     .partition(&mut partition, ())?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Random<R> {
    pub rng: R,
    pub part_count: usize,
}

impl<R> crate::Partition<()> for Random<R>
where
    R: rand::Rng,
{
    type Metadata = ();
    type Error = std::convert::Infallible;

    fn partition(&mut self, part_ids: &mut [usize], _: ()) -> Result<Self::Metadata, Self::Error> {
        for part_id in part_ids {
            *part_id = self.rng.gen_range(0..self.part_count);
        }
        Ok(())
    }
}
