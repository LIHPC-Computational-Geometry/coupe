//! Utilities to handle topologic concepts and metrics related to mesh

use num::FromPrimitive;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use sprs::CsMatView;
use std::collections::HashSet;
use std::iter::Sum;
use std::ops::Mul;

/// The edge cut of a partition.
///
/// Given a partition and a weighted graph associated to a mesh, the edge cut of
/// a partition is defined as the total weight of the edges that link graph
/// nodes of different parts.
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
pub fn edge_cut<T>(adjacency: CsMatView<'_, T>, partition: &[usize]) -> T
where
    T: Copy + Sum + Send + Sync + PartialEq,
{
    let indptr = adjacency.indptr().into_raw_storage();
    let indices = adjacency.indices();
    let data = adjacency.data();
    indptr
        .par_iter()
        .zip(&indptr[1..])
        .enumerate()
        .map(|(node, (start, end))| {
            let neighbors = &indices[*start..*end];
            let edge_weights = &data[*start..*end];
            let node_part = partition[node];
            neighbors
                .iter()
                .zip(edge_weights)
                .take_while(|(neighbor, _edge_weight)| **neighbor < node)
                .filter(|(neighbor, _edge_weight)| node_part != partition[**neighbor])
                .map(|(_neighbor, edge_weight)| *edge_weight)
                .sum()
        })
        .sum()
}

/// Compute the λ-1 cut (lambda-1 cut).
///
/// The λ-1 cut is the sum, for each vertex, of the number of different parts in
/// its neighborhood times its communication weight.
///
/// This metric better represents the actual communication cost of a partition,
/// albeit more expensive to compute.
pub fn lambda_cut<T, W>(adjacency: CsMatView<'_, T>, partition: &[usize], weights: W) -> W::Item
where
    W: IntoParallelIterator,
    W::Iter: IndexedParallelIterator,
    W::Item: Sum + Mul<Output = W::Item> + FromPrimitive,
{
    let indptr = adjacency.indptr().into_raw_storage();
    let indices = adjacency.indices();
    indptr
        .par_iter()
        .zip(&indptr[1..])
        .zip(weights)
        .enumerate()
        .map_with(
            HashSet::new(),
            |neighbor_parts, (node, ((start, end), node_weight))| {
                let neighbors = &indices[*start..*end];
                let node_part = partition[node];
                neighbor_parts.clear();
                for neighbor in neighbors {
                    let neighbor_part = partition[*neighbor];
                    if neighbor_part == node_part {
                        continue;
                    }
                    neighbor_parts.insert(neighbor_part);
                }
                W::Item::from_usize(neighbor_parts.len()).unwrap() * node_weight
            },
        )
        .sum()
}
