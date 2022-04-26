//! Utilities to handle topologic concepts and metrics related to mesh

use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use sprs::CsMat;
use sprs::CsMatView;
use sprs::TriMat;
use std::iter::Sum;

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
pub fn edge_cut<T>(adjacency: CsMatView<T>, partition: &[usize]) -> T
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

/// Build an adjacency matrix from a mesh connectivity.
///
/// # Inputs
///
///  - `conn`: the connectivity of a mesh. It's a sparse matrix in which the `(i, j)` entry is equal to 1
///            if and only if nodes `i` and `j` are linked.
///  - `num_common_nodes`: defines how the adjacency matrix is build. Two elements of the mesh
///                        are linked in the adjacency graph is they have exactly `num_common_nodes` nodes in common.
///
/// # Output
///
/// A sparse matrix in which the enty `(i, j)` is non-zero if and only if
/// the mesh elements `i` and `j` are linked in the mesh graph.
///
/// If the entry `(i, j)` is non-zero, then its value is the weight of the edge between `i`
/// and `j` (default to `1.0`).
pub fn adjacency_matrix(conn: CsMatView<u32>, num_common_nodes: u32) -> CsMat<f64> {
    // currently this matmul operation is very slow
    let graph = &conn * &conn.transpose_view();

    let nnz = graph
        .iter()
        .filter(|(n, _)| **n == num_common_nodes)
        .count();

    let mut row_nnz = Vec::with_capacity(nnz);
    let mut col_nnz = Vec::with_capacity(nnz);
    let mut val_nnz = Vec::with_capacity(nnz);

    for (val, (i, j)) in graph.iter() {
        if *val == num_common_nodes {
            // ret.insert(i, j, 1.);
            row_nnz.push(i);
            col_nnz.push(j);
            val_nnz.push(1.);
        }
    }

    TriMat::from_triplets(graph.shape(), row_nnz, col_nnz, val_nnz).to_csc()
}
