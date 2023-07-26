use super::Error;
use crate::imbalance;
use crate::topology::Topology;
use num_traits::ToPrimitive;
use num_traits::Zero;
use num_traits::{FromPrimitive, Signed};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashSet;
use std::iter::Sum;
use std::marker::PhantomData;
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

macro_rules! impl_signednum_signed {
    ( $($t:ty),* ) => {
    $( impl SignedNum for $t
    {
        type SignedType = Self;

        fn to_signed(self) -> Self { self }
    }) *
    }
}

impl_signednum_signed! {i8, i16, i32, i64, i128, f32, f64}

macro_rules! impl_signednum_unsigned {
    ( $(($t:ty, $s:ty)),* ) => {
    $( impl SignedNum for $t
    {
        type SignedType = $s;

        fn to_signed(self) -> Self::SignedType { self.try_into().unwrap() }
    }) *
    }
}

impl_signednum_unsigned! {(u8,i8), (u16,i16), (u32,i32), (u64,i64), (u128,i128), (usize, isize)}

type VertexId = usize;
type EdgeId = usize;
type PartId = usize;

#[derive(Debug)]
struct TopologicalPart<'a, Adj, T, W>
where
    T: PathWeight,
    <T as SignedNum>::SignedType: Signed + TryFrom<T> + Copy,
    Adj: Topology<T> + Sync,
    W: PathWeight,
{
    part: Vec<PartId>,
    pub cg: Vec<T::SignedType>,
    adjacency: &'a Adj,
    part_loads: Vec<W>,
    weights: &'a [W],
}

impl<'a, Adj, T, W> TopologicalPart<'a, Adj, T, W>
where
    T: PathWeight,
    <T as SignedNum>::SignedType: Signed + TryFrom<T> + Copy,
    Adj: Topology<T> + Sync,
    W: PathWeight,
{
    fn new(topo: &'a Adj, weights: &'a [W], part: &[PartId]) -> Self {
        let cg = Vec::with_capacity(part.len());

        let mut out = Self {
            cg,
            part: Vec::from(part),
            adjacency: topo,
            part_loads: imbalance::compute_parts_load(part, 2, weights.par_iter().cloned()),
            weights,
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
struct Path<'a, Adj, T, W>
where
    T: PathWeight,
    <T as SignedNum>::SignedType: Signed + TryFrom<T> + Copy,
    Adj: Topology<T> + Sync,
    W: PathWeight,
{
    path: Vec<VertexId>,
    last_side: PartId,
    cost: T::SignedType,
    topo_part: &'a TopologicalPart<'a, Adj, T, W>,
    target_load: W,
    part_loads: [W; 2],
}

impl<'a, Adj, T, W> Path<'a, Adj, T, W>
where
    T: PathWeight,
    <T as SignedNum>::SignedType: Signed + TryFrom<T> + Copy,
    Adj: Topology<T> + Sync,
    W: PathWeight,
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
    fn new(topo_part: &'a TopologicalPart<'a, Adj, T, W>) -> Self {
        let part_loads = [topo_part.part_loads[0], topo_part.part_loads[1]];
        let target_load =
            W::from_f64(((part_loads[0] + part_loads[1]).to_f64().unwrap() * 0.005).ceil())
                .unwrap();
        Self {
            path: Vec::new(),
            last_side: 0,
            cost: T::SignedType::zero(),
            topo_part,
            target_load,
            part_loads,
        }
    }

    fn add_to_path(&mut self, (v, cost): (VertexId, T::SignedType)) {
        self.path.push(v);
        self.cost += cost;
        self.last_side = self.topo_part.part[v];
    }

    /// Find best candidate to be swapped, from candidates `iter`
    /// TODO: check imbalance
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

    /// Create an optimization path, beginning in `side`
    /// TODO: try more candidates at beginning
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
pub struct PathOptimization {
}

impl<'a, Adj, T, W> crate::Partition<(Adj, &'a [W])> for PathOptimization
where
    Adj: Topology<T> + Sync + 'a,
    T: PathWeight,
    <T as SignedNum>::SignedType: Signed + TryFrom<T> + Copy,
    W: PathWeight,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (Adj, &'a [W]),
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

        let mut tp = TopologicalPart::new(&adjacency, weights, part_ids);

        let mut side = 0;
        while let Some(p) = Path::new(&tp).find_path(side) {
            eprintln!("found path: {:?}", p.path);
            tp.flip_flop(p.path);
            side = TopologicalPart::<'a, Adj, T, W>::flip_part(side) ;
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

    struct Topo<T>(sprs::CsMat<T>)
    where
        T: PathWeight;

    struct Instance {
        pub geometry: Vec<Point2D>,
        pub v_weights: Vec<f64>,
        pub topology: Topo<usize>,
        pub partition: Vec<usize>,
    }

    impl<T> Topo<T>
    where
        T: PathWeight,
    {
        fn new() -> Self {
            Self(sprs::CsMat::empty(sprs::CSR, 0))
        }

        fn add_edge(&mut self, u: VertexId, v: VertexId, weight: T) {
            self.0.insert(u, v, weight);
            self.0.insert(v, u, weight);
        }
    }

    impl Instance {
        fn create_instance() -> Self {
            let mut out = Self {
                geometry: Vec::with_capacity(8),
                v_weights: vec![1.0; 8],
                partition: Vec::with_capacity(8),
                topology: Topo::new(),
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

            out.topology.add_edge(0, 1, 1);
            out.topology.add_edge(1, 2, 1);
            out.topology.add_edge(2, 3, 1);
            out.topology.add_edge(4, 5, 1);
            out.topology.add_edge(5, 6, 1);
            out.topology.add_edge(6, 7, 1);
            out.topology.add_edge(0, 4, 1);
            out.topology.add_edge(1, 5, 1);
            out.topology.add_edge(2, 6, 1);
            out.topology.add_edge(3, 7, 1);

            out
        }
    }
    #[test]
    fn check_cg() {
        let instance = Instance::create_instance();

        let topo = instance.topology.0.view();
        let mut tp = TopologicalPart::new(
            &topo,
            instance.v_weights.as_slice(),
            instance.partition.as_slice(),
        );
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
