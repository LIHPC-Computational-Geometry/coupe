use mesh_io::medit::Mesh;

/// The adjacency matrix that model the dual graph of the given mesh.
pub fn dual(mesh: &Mesh) -> sprs::CsMat<f64> {
    let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
    let elements = || {
        mesh.elements()
            .filter(|(el_type, _nodes, _ref)| el_type.dimension() == mesh.dimension())
            .map(|(_el_type, nodes, _ref)| nodes)
            .enumerate()
    };
    for (e1, e1_nodes) in elements() {
        for (e2, e2_nodes) in elements() {
            if e1 == e2 {
                continue;
            }
            let nodes_in_common = e1_nodes
                .iter()
                .filter(|e1_node| e2_nodes.contains(e1_node))
                .count();
            let are_neighbors = mesh.dimension() <= nodes_in_common;
            if are_neighbors {
                adjacency.insert(e1, e2, 1.0);
            }
        }
    }

    adjacency
}
