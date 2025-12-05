use num_traits::FromPrimitive;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator as _;
use std::collections::HashSet;
use std::iter::Sum;
use std::ops::Mul;

#[cfg(feature = "sprs")]
mod sprs;

/// `Topology` is implemented for types that represent mesh topology.
pub trait Topology<E> {
    /// Return type for [`Topology::neighbors`].
    ///
    /// This is an implementation detail and will be removed when Rust allows us
    /// to do so (at most when async fns are allowed in traits).
    type Neighbors<'n>: Iterator<Item = (usize, E)>
    where
        Self: 'n;

    /// The number of elements in the mesh.
    fn len(&self) -> usize;

    /// Whether the topology has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An iterator over the neighbors of the given vertex.
    fn neighbors(&self, vertex: usize) -> Self::Neighbors<'_>;

    /// The edge cut of a partition.
    ///
    /// Given a partition and a weighted graph associated to a mesh, the edge
    /// cut of a partition is defined as the total weight of the edges that link
    /// graph nodes of different parts.
    ///
    /// # Example
    ///
    /// A partition with two parts (0 and 1)
    /// ```text,ignore
    ///          0
    ///    1*──┆─*────* 0
    ///    ╱ ╲ ┆╱    ╱
    ///  1*  1*┆ <┈┈╱┈┈┈ Dotted line passes through edged that contribute to edge cut.
    ///    ╲ ╱ ┆   ╱     If all edges have a weight of 1 then edge_cut = 3
    ///    1*  ┆╲ ╱
    ///          * 0
    /// ```
    fn edge_cut(&self, partition: &[usize]) -> E
    where
        Self: Sync,
        E: Sum + Send,
    {
        (0..self.len())
            .into_par_iter()
            .map(|vertex| {
                let vertex_part = partition[vertex];
                self.neighbors(vertex)
                    .filter(|(neighbor, _edge_weight)| {
                        vertex_part != partition[*neighbor] && *neighbor < vertex
                    })
                    .map(|(_neighbor, edge_weight)| edge_weight)
                    .sum()
            })
            .sum()
    }

    /// The λ-1 cut (lambda-1 cut) of a partition.
    ///
    /// The λ-1 cut is the sum, for each vertex, of the number of different
    /// parts in its neighborhood times its communication weight.
    ///
    /// This metric better represents the actual communication cost of a
    /// partition, albeit more expensive to compute.
    fn lambda_cut<W>(&self, partition: &[usize], weights: W) -> W::Item
    where
        Self: Sync,
        W: IntoParallelIterator,
        W::Iter: IndexedParallelIterator,
        W::Item: Sum + Mul<Output = W::Item> + FromPrimitive,
    {
        (0..self.len())
            .into_par_iter()
            .zip(weights)
            .map_with(HashSet::new(), |neighbor_parts, (vertex, vertex_weight)| {
                neighbor_parts.clear();
                neighbor_parts.insert(partition[vertex]);
                neighbor_parts.extend(self.neighbors(vertex).map(|(v, _)| partition[v]));
                W::Item::from_usize(neighbor_parts.len() - 1).unwrap() * vertex_weight
            })
            .sum()
    }
}

impl<'a, T, E> Topology<E> for &'a T
where
    E: Copy,
    T: Topology<E>,
{
    type Neighbors<'n>
        = T::Neighbors<'n>
    where
        Self: 'n;

    fn len(&self) -> usize {
        T::len(self)
    }

    fn neighbors(&self, vertex: usize) -> Self::Neighbors<'_> {
        T::neighbors(self, vertex)
    }
}
