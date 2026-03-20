use super::Topology;
use num_traits::FromPrimitive;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashSet;
use std::iter::Cloned;
use std::iter::Sum;
use std::iter::Zip;
use std::ops::Mul;

impl<'a, E> Topology<E> for sprs::CsMatView<'a, E>
where
    E: Copy + Sync,
{
    type Neighbors<'n>
        = Zip<Cloned<std::slice::Iter<'n, usize>>, Cloned<std::slice::Iter<'n, E>>>
    where
        Self: 'n;

    fn len(&self) -> usize {
        debug_assert_eq!(self.rows(), self.cols());
        self.rows()
    }

    fn neighbors(&self, vertex: usize) -> Self::Neighbors<'_> {
        // `CsVecView` does not implement `IntoIterator`, so we have to
        // implement it ourselves. It's needed to pass through the `&'_ self`
        // lifetime and not end up with a local one.
        let (indices, data) = self.outer_view(vertex).unwrap().into_raw_storage();
        indices.iter().cloned().zip(data.iter().cloned())
    }

    fn edge_cut(&self, partition: &[usize]) -> E
    where
        E: Sum + Send,
    {
        let indptr = self.indptr().into_raw_storage();
        let indices = self.indices();
        let data = self.data();
        indptr
            .par_iter()
            .zip(&indptr[1..])
            .enumerate()
            .map(|(vertex, (start, end))| {
                let neighbors = &indices[*start..*end];
                let edge_weights = &data[*start..*end];
                let vertex_part = partition[vertex];
                neighbors
                    .iter()
                    .zip(edge_weights)
                    .take_while(|(neighbor, _edge_weight)| **neighbor < vertex)
                    .filter(|(neighbor, _edge_weight)| vertex_part != partition[**neighbor])
                    .map(|(_neighbor, edge_weight)| *edge_weight)
                    .sum()
            })
            .sum()
    }

    fn lambda_cut<W>(&self, partition: &[usize], weights: W) -> W::Item
    where
        W: IntoParallelIterator,
        W::Iter: IndexedParallelIterator,
        W::Item: Sum + Mul<Output = W::Item> + FromPrimitive,
    {
        let indptr = self.indptr().into_raw_storage();
        let indices = self.indices();
        indptr
            .par_iter()
            .zip(&indptr[1..])
            .zip(weights)
            .enumerate()
            .map_with(
                HashSet::new(),
                |neighbor_parts, (vertex, ((start, end), vertex_weight))| {
                    let neighbors = &indices[*start..*end];
                    neighbor_parts.clear();
                    neighbor_parts.insert(partition[vertex]);
                    neighbor_parts.extend(neighbors.iter().map(|v| partition[*v]));
                    W::Item::from_usize(neighbor_parts.len() - 1).unwrap() * vertex_weight
                },
            )
            .sum()
    }
}
