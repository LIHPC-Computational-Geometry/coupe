use super::Error;
use crate::topology::Topology;
use num_traits::ToPrimitive;
use num_traits::Zero;
use num_traits::{FromPrimitive, Signed};
use std::collections::HashSet;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;

/// Trait alias for values accepted as weights by [PathOptimization].
pub trait PathWeight:
    Copy
    + std::fmt::Debug
    + Send
    + Sync
    + Sum
    + PartialOrd
    + FromPrimitive
    + ToPrimitive
    + Zero
    + Sub<Output = Self>
    + AddAssign
    + SubAssign
    + SignedNum
{
}

impl<T> PathWeight for T
where
    Self: Copy + std::fmt::Debug + Send + Sync,
    Self: Sum + PartialOrd + FromPrimitive + ToPrimitive + Zero,
    Self: Sub<Output = Self> + AddAssign + SubAssign,
    Self: SignedNum,
{
}

pub trait SignedNum: Sized {
    type SignedType: Copy + PartialOrd + Zero + AddAssign;

    fn to_signed(self) -> Self::SignedType;
}
//
// impl<T: Signed + Copy + PartialOrd + Zero + AddAssign > SignedNum for T {
//     type SignedType = T;
//
//     fn to_signed(self) -> T {
//         self
//     }
// }

impl SignedNum for usize {
    type SignedType = i64;

    fn to_signed(self) -> Self::SignedType {
        self.try_into().unwrap()
    }
}

impl SignedNum for i64 {
    type SignedType = i64;

    fn to_signed(self) -> Self::SignedType {
        self
    }
}

impl SignedNum for f64 {
    type SignedType = f64;

    fn to_signed(self) -> Self::SignedType {
        self
    }
}

type VertexId = usize;
type EdgeId = usize;
type PartId = usize;

#[derive(Debug)]
struct TopologicalPart<'a, Adj, T>
where
    T: PathWeight,
    <T as SignedNum>::SignedType: Signed + TryFrom<T> + Copy,
    Adj: Topology<T> + Sync,
{
    part: Vec<PartId>,
    pub cg: Vec<T::SignedType>,
    adjacency: &'a Adj,
}

impl<'a, Adj, T> TopologicalPart<'a, Adj, T>
where
    T: PathWeight,
    <T as SignedNum>::SignedType: Signed + TryFrom<T> + Copy,
    Adj: Topology<T> + Sync,
{
    fn new(topo: &'a Adj, part: &[PartId]) -> Self {
        let cg = Vec::with_capacity(part.len());

        let mut out = Self {
            cg,
            part: Vec::from(part),
            adjacency: topo,
        };

        (0..part.len()).for_each(|v| out.cg.push(out.compute_cg(v)));
        out
    }

    fn flip_part(part: PartId) -> PartId {
        1 - part
    }

    fn compute_cg(&self, v: VertexId) -> T::SignedType {
        self.adjacency
            .neighbors(v)
            .fold(T::SignedType::zero(), |acc, (neighbor, edge_weight)| {
                if self.part[neighbor] == self.part[v] {
                    acc + edge_weight.to_signed()
                } else {
                    acc - edge_weight.to_signed()
                }
            })
    }

    fn flip_flop<P>(&mut self, path: P)
    where
        P: IntoIterator<Item = VertexId>,
    {
        let cg_to_update = path
            .into_iter()
            .fold(HashSet::<VertexId>::new(), |mut acc, v| {
                self.part[v] = Self::flip_part(self.part[v]);
                acc.insert(v);
                acc.extend(self.adjacency.neighbors(v).map(|(w, _)| w));
                acc
            });
        cg_to_update.iter().for_each(|&v| {
            self.cg[v] = self.compute_cg(v);
        });
    }
}

#[derive(Debug)]
struct Path<'a, Adj, T>
where
    T: PathWeight,
    <T as SignedNum>::SignedType: Signed + TryFrom<T> + Copy,
    Adj: Topology<T> + Sync,
{
    path: Vec<VertexId>,
    last_side: PartId,
    cost: T::SignedType,
    topo_part: &'a TopologicalPart<'a, Adj, T>,
}

impl<'a, Adj, T> Path<'a, Adj, T>
where
    T: PathWeight,
    <T as SignedNum>::SignedType: Signed + TryFrom<T> + Copy,
    Adj: Topology<T> + Sync,
{
    /// Compute whether a vertex v is suitable to extend P
    /// Not for hypergraph yet.
    fn flip_cost_incr(&self, v: usize) -> T::SignedType {
        let (e_nc, e_c) = self.topo_part.adjacency.neighbors(v).fold(
            (T::SignedType::zero(), T::SignedType::zero()),
            |acc, (neighbor, edge_weight)| {
                if !self.path.contains(&neighbor) {
                    return acc;
                }
                if self.topo_part.part[neighbor] == self.topo_part.part[v] {
                    (acc.0 + edge_weight.to_signed(), acc.1)
                } else {
                    (acc.0, acc.1 + edge_weight.to_signed())
                }
            },
        );
        self.topo_part.cg[v] + (e_c - e_nc) + (e_c - e_nc) //* T::from(2)
    }

    /// Find the next path vertex
    fn select_next_cell(&self) -> Option<(usize, T::SignedType)> {
        let side = (1 - self.last_side) as usize;
        // Take the second last recent because "minimization"
        let v = self.path[self.path.len() - 2];
        self.topo_part
            .adjacency
            .neighbors(v)
            .filter_map(|(neighbor, _edge_weight)| {
                if self.topo_part.part[neighbor] != side {
                    return None;
                };
                if self.path.contains(&neighbor) {
                    return None;
                }
                // Check if move decreases cut
                let cost = self.flip_cost_incr(neighbor);
                if cost >= T::SignedType::zero() {
                    return None;
                }
                Some((neighbor, cost))
            })
            .next()
    }

    /// Create an empty path
    fn new(topo_part: &'a TopologicalPart<'a, Adj, T>) -> Self {
        Self {
            path: Vec::new(),
            last_side: 0,
            cost: T::SignedType::zero(),
            topo_part,
        }
    }

    fn add_to_path(&mut self, (v, cost): (VertexId, T::SignedType)) {
        self.path.push(v);
        self.cost += cost;
        self.last_side = self.topo_part.part[v];
    }

    fn find_best<I>(&self, side: usize, iter: I) -> Option<VertexId>
    where
        I: IntoIterator<Item = VertexId>,
    {
        // Should use priority queue
        iter.into_iter()
            .filter(|&v| self.topo_part.part[v] == side)
            .fold(
                (None as Option<VertexId>, T::SignedType::zero()),
                |(id_min, val_min), id| {
                    if id_min.is_none() || self.topo_part.cg[id] <= val_min {
                        (Some(id), self.topo_part.cg[id])
                    } else {
                        (id_min, val_min)
                    }
                },
            )
            .0
    }

    /// Create an optimization path, beginning in side
    fn find_path(self, side: usize) -> Option<Self> {
        // Choose v as the best vertex
        let v = if let Some(v) = self.find_best(side, 0..self.topo_part.part.len()) {
            v
        } else {
            return None;
        };
        // Find w
        let w = if let Some(w) = self.find_best(
            1 - side,
            (0..self.topo_part.part.len()).filter(|&candidate| {
                !self
                    .topo_part
                    .adjacency
                    .neighbors(v)
                    .any(|(neighbor, _)| neighbor == candidate)
            }),
        ) {
            w
        } else {
            return None;
        };

        let mut path = self;
        path.add_to_path((v, path.topo_part.cg[v]));
        path.add_to_path((w, path.topo_part.cg[w]));

        while let Some(candidate) = path.select_next_cell() {
            path.add_to_path(candidate);
        }
        if path.cost < T::SignedType::zero() {
            Some(path)
        } else {
            None
        }
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
pub struct PathOptimization {}

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

        let mut tp = TopologicalPart::new(&adjacency, part_ids);

        let mut side = 0;
        while let Some(p) = Path::new(&tp).find_path(side) {
            tp.flip_flop(p.path);
            side = 1 - side;
        }

        part_ids
            .iter_mut()
            .zip(tp.part)
            .for_each(|(dst, src)| *dst = src);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point2D;
    

    struct Instance {
        pub geometry: Vec<Point2D>,
        pub v_weights: Vec<f64>,
        pub topology: sprs::CsMat<usize>,
        pub partition: Vec<usize>,
    }

    impl Instance {
        fn create_instance() -> Self {
            let mut out = Self {
                geometry: Vec::with_capacity(8),
                v_weights: vec![1.0; 8],
                partition: Vec::with_capacity(8),
                topology: sprs::CsMat::empty(sprs::CSR, 0),
            };

            //    swap
            // 0  1  0  1
            // +--+--+--+
            // |  |  |  |
            // +--+--+--+
            // 0  0  1  1
            out.geometry.extend([
                Point2D::new(0., 0.),
                Point2D::new(1., 0.),
                Point2D::new(2., 0.),
                Point2D::new(3., 0.),
                Point2D::new(0., 1.),
                Point2D::new(1., 1.),
                Point2D::new(2., 1.),
                Point2D::new(3., 1.),
            ]);
            out.partition.extend([0, 0, 1, 1, 0, 1, 0, 1]);

            out.topology.insert(0, 1, 1);
            out.topology.insert(1, 2, 1);
            out.topology.insert(2, 3, 1);
            out.topology.insert(4, 5, 1);
            out.topology.insert(5, 6, 1);
            out.topology.insert(6, 7, 1);
            out.topology.insert(0, 4, 1);
            out.topology.insert(1, 5, 1);
            out.topology.insert(2, 6, 1);
            out.topology.insert(3, 7, 1);

            // symmetry
            out.topology.insert(1, 0, 1);
            out.topology.insert(2, 1, 1);
            out.topology.insert(3, 2, 1);
            out.topology.insert(5, 4, 1);
            out.topology.insert(6, 5, 1);
            out.topology.insert(7, 6, 1);
            out.topology.insert(4, 0, 1);
            out.topology.insert(5, 1, 1);
            out.topology.insert(6, 2, 1);
            out.topology.insert(7, 3, 1);

            out
        }
    }
    #[test]
    fn check_cg() {
        let instance = Instance::create_instance();

        let topo = instance.topology.view();
        let mut tp = TopologicalPart::new(&topo, instance.partition.as_slice());
        println!("CG = {:?}", tp.cg);

        let mut side = 0;
        while let Some(p) = Path::new(&tp).find_path(side) {
            println!("path = {:?}", p);
            tp.flip_flop(p.path);
            side = 1 - side;
            println!("tp = {:?}", tp);
        }
    }
}
