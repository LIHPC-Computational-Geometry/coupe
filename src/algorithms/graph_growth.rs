use crate::ProcessUniqueId;
use rand::seq::SliceRandom;
use sprs::CsMatView;

pub fn graph_growth(
    weights: &[f64],
    adjacency: CsMatView<f64>,
    num_parts: usize,
) -> Vec<ProcessUniqueId> {
    let (shape_x, shape_y) = adjacency.shape();
    assert_eq!(shape_x, shape_y);
    assert_eq!(weights.len(), shape_x);

    let dummy_id = ProcessUniqueId::new();
    let mut initial_ids = vec![dummy_id; weights.len()];
    // let total_weight = weights.iter().sum::<f64>();
    // let weight_per_part = total_weight / num_parts as f64;
    let max_expansion_per_pass = 20;

    let mut rng = rand::thread_rng();

    // select two random nodes to grow from
    let indices = (0..weights.len()).collect::<Vec<_>>();
    let indices = indices
        .as_slice()
        .choose_multiple(&mut rng, num_parts)
        .collect::<Vec<_>>();

    // tracks if each node has already been assigned to a partition or not
    let mut assigned = vec![false; weights.len()];
    let unique_ids = indices
        .iter()
        .map(|_| ProcessUniqueId::new())
        .collect::<Vec<_>>();

    // assign initial nodes
    for (idx, id) in indices.iter().cloned().zip(unique_ids.iter()) {
        initial_ids[*idx] = *id;
        assigned[*idx] = true;
    }

    let mut remaining_nodes = weights.len() - num_parts;

    while remaining_nodes > 0 {
        let mut num_expansion = vec![0; unique_ids.len()];

        for (i, row) in adjacency.outer_iterator().enumerate() {
            let id = initial_ids[i];
            if assigned[i] {
                for (j, _w) in row.iter() {
                    if !assigned[j] {
                        if initial_ids[j] != id {
                            let idx = unique_ids.iter().position(|el| *el == id).unwrap();
                            if num_expansion[idx] > max_expansion_per_pass {
                                break;
                            }
                            num_expansion[idx] += 1;
                            initial_ids[j] = id;
                            assigned[j] = true;
                            remaining_nodes -= 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    initial_ids
}
