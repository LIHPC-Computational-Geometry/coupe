//! Utilities to handle topologic concepts and metrics related to mesh

use crate::ProcessUniqueId;

use sprs::{CsMat, CsMatView};

/// Computes the cutsize of a partition.
///
/// Given a partition and a weighted graph associated to a mesh, the cutsize
/// of a partition woth two parts is defined as the total weights of edged that link
/// two graph nodes of different parts.
///
/// # Example
///
/// A partition with two parts (0 and 1)
/// ```ignore
///          0
///    1*──┆─*────* 0          
///    ╱ ╲ ┆╱    ╱           
///  1*  1*┆ <┈┈╱┈┈┈ Dotted line passes through edged that contribute to cutsize.         
///    ╲ ╱ ┆   ╱     If all edges have a weight of 1 then cutsize = 3         
///    1*  ┆╲ ╱            
///          * 0
/// ```
pub fn cut_size(adjacency: CsMatView<f64>, partition: &[ProcessUniqueId]) -> f64 {
    let mut cut_size = 0.;
    for (i, row) in adjacency.outer_iterator().enumerate() {
        for (j, w) in row.iter() {
            // graph edge are present twice in the matrix be cause of symetry
            if j >= i {
                continue;
            }
            if partition[i] != partition[j] {
                cut_size += w;
            }
        }
    }
    cut_size
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
    let graph = &conn * &conn.transpose_view();
    let mut ret = CsMat::zero(graph.shape());
    let nnz = graph
        .iter()
        .filter(|(n, _)| **n == num_common_nodes)
        .count();
    ret.reserve_nnz(nnz);

    for (val, (i, j)) in graph.iter() {
        if *val == num_common_nodes {
            ret.insert(i, j, 1.);
        }
    }

    ret
}