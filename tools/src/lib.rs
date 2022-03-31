use mesh_io::medit::Mesh;

/// The adjacency matrix that models the dual graph of the given mesh.
pub fn dual(mesh: &Mesh) -> sprs::CsMat<f64> {
    let dimension = mesh.dimension();

    let elements = || {
        mesh.elements()
            .filter(|(el_type, _nodes, _ref)| el_type.dimension() == dimension)
            .map(|(_el_type, nodes, _ref)| nodes)
            .enumerate()
    };

    let element_to_nodes = {
        // To speed up node lookup, we store topology information in a more
        // compact array of element chunks.  Chunks store the nodes of elements
        // of the same type, and their start offset.
        struct ElementChunk<'a> {
            start_idx: usize,
            node_per_element: usize,
            nodes: &'a [usize],
        }
        let topology: Vec<ElementChunk> = mesh
            .topology()
            .iter()
            .filter(|(el_type, _nodes, _refs)| el_type.dimension() == dimension)
            .scan(0, |start_idx, (el_type, nodes, _refs)| {
                let item = ElementChunk {
                    start_idx: *start_idx,
                    node_per_element: el_type.node_count(),
                    nodes,
                };
                *start_idx += nodes.len() / item.node_per_element;
                Some(item)
            })
            .collect();
        move |e: usize| -> &[usize] {
            for item in &topology {
                let e = (e - item.start_idx) * item.node_per_element;
                if e < item.nodes.len() {
                    return &item.nodes[e..e + item.node_per_element];
                }
            }
            unreachable!();
        }
    };

    let mut node_to_elements = vec![Vec::new(); mesh.node_count()];
    for (e, nodes) in elements() {
        for node in nodes {
            let node_elements = &mut node_to_elements[*node];
            if node_elements.is_empty() {
                node_elements.reserve(8);
            }
            if let Err(idx) = node_elements.binary_search(&e) {
                node_elements.insert(idx, e);
            }
        }
    }

    let mut adjacency = sprs::CsMat::empty(sprs::CSR, mesh.element_count());
    adjacency.reserve_nnz(mesh.element_count());

    for (e1, e1_nodes) in elements() {
        let neighbors = e1_nodes
            .iter()
            .flat_map(|node| &node_to_elements[*node])
            .cloned();
        for e2 in neighbors {
            if e1 == e2 {
                continue;
            }
            let e2_nodes = element_to_nodes(e2);
            let nodes_in_common = e1_nodes
                .iter()
                .filter(|e1_node| e2_nodes.contains(e1_node))
                .count();
            let are_neighbors = dimension <= nodes_in_common;
            if are_neighbors {
                adjacency.insert(e1, e2, 1.0);
            }
        }
    }

    adjacency
}
