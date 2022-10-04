use std::fmt;

mod arc_swap;
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

pub use arc_swap::ArcSwap;
pub use arc_swap::AsWeight;
pub use arc_swap::Metadata as AsMetadata;
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
pub use hilbert_curve::ZCurve;
pub use k_means::KMeans;
pub use kernighan_lin::KernighanLin;
pub use kk::KarmarkarKarp;
pub use kk::KkWeight;
pub use multi_jagged::MultiJagged;
pub use recursive_bisection::Rcb;
pub use recursive_bisection::RcbWeight;
pub use recursive_bisection::Rib;
pub use vn::VnBest;
pub use vn::VnBestWeight;
pub use vn::VnFirst;
pub use vn::VnFirstWeight;

/// Common errors thrown by algorithms.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Error {
    /// No partition that matches the given criteria could been found.
    NotFound,

    /// Input sets don't have matching lengths.
    InputLenMismatch { expected: usize, actual: usize },

    /// Input contains negative values and such values are not supported.
    NegativeValues,

    /// When a partition improving algorithm is given more than 2 parts.
    BiPartitioningOnly,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFound => write!(f, "no partition found"),
            Error::InputLenMismatch { expected, actual } => write!(
                f,
                "input sets don't have the same length (expected {expected} items, got {actual})",
            ),
            Error::NegativeValues => write!(f, "input contains negative values"),
            Error::BiPartitioningOnly => write!(f, "expected no more than two parts"),
        }
    }
}

impl std::error::Error for Error {}

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
