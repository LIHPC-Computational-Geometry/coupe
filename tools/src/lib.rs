use mesh_io::medit::Mesh;

pub fn adjacency(mesh: &Mesh) -> sprs::CsMat<f64> {
    let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
    for (e1, (e1_type, e1_nodes, _e1_ref)) in mesh.elements().enumerate() {
        if e1_type.dimension() != mesh.dimension() {
            continue;
        }
        for (e2, (e2_type, e2_nodes, _e2_ref)) in mesh.elements().enumerate() {
            if e1 == e2 || e2_type.dimension() != mesh.dimension() {
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
