use super::Error;
use crate::topology::Topology;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use num_traits::Zero;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::iter::Sum;
use std::ops::Sub;
use std::ops::SubAssign;
use std::ops::{AddAssign, Index};

/// Trait alias for values accepted as weights by [PathOptimization].
pub trait PathWeight
where
    Self: Copy + std::fmt::Debug + Send + Sync,
    Self: Sum + PartialOrd + FromPrimitive + ToPrimitive + Zero,
    Self: Sub<Output = Self> + AddAssign + SubAssign,
{
}

impl<T> PathWeight for T
where
    Self: Copy + std::fmt::Debug + Send + Sync,
    Self: Sum + PartialOrd + FromPrimitive + ToPrimitive + Zero,
    Self: Sub<Output = Self> + AddAssign + SubAssign,
{
}

struct Gains<T: PathWeight> {
    edge_sum: Vec<[T; 2]>,
    parts: Vec<usize>,
}

impl<T: PathWeight> Gains<T> {
    fn gain(&self, vertex: usize) -> T {
        self.edge_sum[vertex][1 - self.parts[vertex]] - self.edge_sum[vertex][self.parts[vertex]]
    }
}

struct Path<Adj, T>
where
    T: PathWeight,
    Adj: Topology<T> + Sync,
{
    path: Vec<usize>,
    part: Vec<usize>,
    cg: Vec<T>,
    last_side: u32,
    adjacency: Adj,
}

impl<Adj, T> Path<Adj, T>
where
    T: PathWeight,
    Adj: Topology<T> + Sync,
{
    /// Compute whether a vertex v is suitable to extend P
    /// Not for hypergraph yet.
    fn flip_cost_incr(&self, v: usize) -> T {
        let (e_nc, e_c) = self.adjacency.neighbors(v).fold(
            (T::zero(), T::zero()),
            |acc, (neighbor, edge_weight)| {
                if !self.path.contains(&neighbor) {
                    return acc;
                }
                if self.part[neighbor] == self.part[v] {
                    (acc.0 + edge_weight, acc.1)
                } else {
                    (acc.0, acc.1 + edge_weight)
                }
            },
        );
        self.cg[v] + (e_c - e_nc) + (e_c - e_nc) //* T::from(2)
    }

    /// Find the next path vertex
    fn select_next_cell(&self) -> Option<usize> {
        let side = (1 - self.last_side) as usize;
        let v = self.path[self.path.len() - 2];
        self.adjacency
            .neighbors(v)
            .filter_map(|(neighbor, _edge_weight)| {
                if self.part[neighbor] != side {
                    return None;
                };
                if self.path.contains(&neighbor) {
                    return None;
                }
                if self.flip_cost_incr(neighbor) >= T::zero() {
                    return None;
                }
                Some(neighbor)
            })
            .next()
    }


}

/// Path Optimization
///
/// An implementation of the Path Optimization algorithm
/// for graph partitioning. This implementation handles only two parts.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), coupe::Error> {
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// //    swap
/// // 0  1  0  1
/// // +--+--+--+
/// // |  |  |  |
/// // +--+--+--+
/// // 0  0  1  1
/// let points = [
///     Point2D::new(0., 0.),
///     Point2D::new(1., 0.),
///     Point2D::new(2., 0.),
///     Point2D::new(3., 0.),
///     Point2D::new(0., 1.),
///     Point2D::new(1., 1.),
///     Point2D::new(2., 1.),
///     Point2D::new(3., 1.),
/// ];
/// let weights = [1.0; 8];
/// let mut partition = [0, 0, 1, 1, 0, 1, 0, 1];
///
/// let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
/// adjacency.insert(0, 1, 1);
/// adjacency.insert(1, 2, 1);
/// adjacency.insert(2, 3, 1);
/// adjacency.insert(4, 5, 1);
/// adjacency.insert(5, 6, 1);
/// adjacency.insert(6, 7, 1);
/// adjacency.insert(0, 4, 1);
/// adjacency.insert(1, 5, 1);
/// adjacency.insert(2, 6, 1);
/// adjacency.insert(3, 7, 1);
///
/// // symmetry
/// adjacency.insert(1, 0, 1);
/// adjacency.insert(2, 1, 1);
/// adjacency.insert(3, 2, 1);
/// adjacency.insert(5, 4, 1);
/// adjacency.insert(6, 5, 1);
/// adjacency.insert(7, 6, 1);
/// adjacency.insert(4, 0, 1);
/// adjacency.insert(5, 1, 1);
/// adjacency.insert(6, 2, 1);
/// adjacency.insert(7, 3, 1);
///
/// // Set the imbalance tolerance to 25% to provide enough room for FM to do
/// // the swap.
/// coupe::FiducciaMattheyses { max_imbalance: Some(0.25), ..Default::default() }
///     .partition(&mut partition, (adjacency.view(), &weights))?;
///
/// assert_eq!(partition, [0, 0, 1, 1, 0, 0, 1, 1]);
/// # Ok(())
/// # }
/// ```
///
/// # Reference
///
/// Berry, Jonathan W, and Mark K Goldberg. “Path Optimization for Graph Partitioning Problems.”
/// Discrete Applied Mathematics 90, no. 1–3 (January 1999): 27–50.
/// https://doi.org/10.1016/S0166-218X(98)00084-5.
///
#[derive(Debug, Clone, Copy, Default)]
pub struct PathOptimization {
    /// If `Some(max)` then the algorithm will not do more than `max` passes.
    /// If `None` then it will stop on the first non-fruitful pass.
    pub max_passes: Option<usize>,

    /// If `Some(max)` then the algorithm will not do more than `max` moves per
    /// pass.  If `None` then passes will stop when no more vertices yield a
    /// positive gain, and no more bad moves can be made.
    pub max_moves_per_pass: Option<usize>,

    /// If `Some(max)` then the algorithm will not move vertices in ways that
    /// the imbalance goes over `max`.  If `None`, then it will default to the
    /// imbalance of the input partition.
    pub max_imbalance: Option<f64>,

    /// How many moves that yield negative gains can be made before a pass ends.
    pub max_bad_move_in_a_row: usize,
}

impl<'a, T, W> crate::Partition<(T, &'a [W])> for PathOptimization
where
    T: Topology<i64> + Sync,
    W: PathWeight,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (T, &'a [W]),
    ) -> Result<Self::Metadata, Self::Error> {
        if part_ids.is_empty() {
            return Ok(());
        }
        if part_ids.len() != weights.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: weights.len(),
            });
        }
        if part_ids.len() != adjacency.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: adjacency.len(),
            });
        }
        if 1 < *part_ids.iter().max().unwrap_or(&0) {
            return Err(Error::BiPartitioningOnly);
        }
        Ok(())
    }
}
