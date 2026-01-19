use rand::seq::IndexedRandom;
use sprs::CsMatView;

fn graph_growth(
    initial_ids: &mut [usize],
    weights: &[f64],
    adjacency: CsMatView<'_, f64>,
    num_parts: usize,
) {
    let (shape_x, shape_y) = adjacency.shape();
    assert_eq!(shape_x, shape_y);
    assert_eq!(weights.len(), shape_x);

    // let total_weight = weights.iter().sum::<f64>();
    // let weight_per_part = total_weight / num_parts as f64;
    let max_expansion_per_pass = 20;

    let mut rng = rand::rng();

    // select two random nodes to grow from
    let indices = (0..weights.len()).collect::<Vec<_>>();
    let indices = indices
        .as_slice()
        .choose_multiple(&mut rng, num_parts)
        .cloned()
        .collect::<Vec<_>>();

    // tracks if each node has already been assigned to a partition or not
    let mut assigned = vec![false; weights.len()];
    let unique_ids = indices.clone();

    // assign initial nodes
    for (idx, id) in indices.iter().zip(unique_ids.iter()) {
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
                    if !assigned[j] && initial_ids[j] != id {
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

/// Graph Growth algorithm
///
/// A topologic algorithm that generates a partition from a topologic mesh.
/// Given a number k of parts, the algorithm selects k nodes randomly and assigns them to a different part.
/// Then, at each iteration, each part is expanded to neighbor nodes that are not yet assigned to a part
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), std::convert::Infallible> {
/// use coupe::Partition as _;
/// use coupe::Point2D;
/// use sprs::CsMat;
///
/// // +--+--+--+
/// // |  |  |  |
/// // +--+--+--+
///
///  let weights = [1.0; 8];
///  let mut partition = [0; 8];
///
///  let mut adjacency = CsMat::empty(sprs::CSR, 8);
///  adjacency.reserve_outer_dim(8);
///  eprintln!("shape: {:?}", adjacency.shape());
///  adjacency.insert(0, 1, 1.);
///  adjacency.insert(1, 2, 1.);
///  adjacency.insert(2, 3, 1.);
///  adjacency.insert(4, 5, 1.);
///  adjacency.insert(5, 6, 1.);
///  adjacency.insert(6, 7, 1.);
///  adjacency.insert(0, 4, 1.);
///  adjacency.insert(1, 5, 1.);
///  adjacency.insert(2, 6, 1.);
///  adjacency.insert(3, 7, 1.);
///  
///  // symmetry
///  adjacency.insert(1, 0, 1.);
///  adjacency.insert(2, 1, 1.);
///  adjacency.insert(3, 2, 1.);
///  adjacency.insert(5, 4, 1.);
///  adjacency.insert(6, 5, 1.);
///  adjacency.insert(7, 6, 1.);
///  adjacency.insert(4, 0, 1.);
///  adjacency.insert(5, 1, 1.);
///  adjacency.insert(6, 2, 1.);
///  adjacency.insert(7, 3, 1.);
///
/// coupe::GraphGrowth { part_count: 2 }
///     .partition(&mut partition, (adjacency.view(), &weights))?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GraphGrowth {
    pub part_count: usize,
}

impl<'a, W> crate::Partition<(CsMatView<'a, f64>, W)> for GraphGrowth
where
    W: AsRef<[f64]>,
{
    type Metadata = ();
    type Error = std::convert::Infallible;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (CsMatView<'_, f64>, W),
    ) -> Result<Self::Metadata, Self::Error> {
        graph_growth(
            part_ids,
            weights.as_ref(),
            adjacency.view(),
            self.part_count,
        );
        Ok(())
    }
}
