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

pub use ckk::CompleteKarmarkarKarp;
pub use fiduccia_mattheyses::FiducciaMattheyses;
pub use graph_growth::GraphGrowth;
pub use greedy::Greedy;
pub use hilbert_curve::Error as HilbertCurveError;
pub use hilbert_curve::HilbertCurve;
pub use k_means::KMeans;
pub use kernighan_lin::KernighanLin;
pub use kk::KarmarkarKarp;
pub use multi_jagged::MultiJagged;
pub use recursive_bisection::Rcb;
pub use recursive_bisection::Rib;
pub use vn::VnBest;
pub use vn::VnFirst;
pub use z_curve::ZCurve;

/// Common errors thrown by algorithms.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// No partition that matches the given criteria could been found.
    NotFound,

    /// Input sets don't have matching lengths.
    InputLenMismatch { expected: usize, actual: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFound => write!(f, "no partition found"),
            Error::InputLenMismatch { expected, actual } => write!(
                f,
                "input sets don't have the same length (expected {expected} items, got {actual})",
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Map elements to parts randomly.
///
/// # Example
///
/// ```rust
/// use coupe::Partition as _;
/// use rand;
///
/// let mut partition = [0; 12];
///
/// coupe::Random { rng: rand::thread_rng(), part_count: 3 }
///     .partition(&mut partition, ())
///     .unwrap();
/// ```
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
