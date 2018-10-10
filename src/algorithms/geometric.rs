use geometry::Point2D;
use itertools::{self, Itertools};
use snowflake::ProcessUniqueId;
use std::cmp::Ordering;

/// # Recursive Coordinate Bisection algorithm
/// Partitions a mesh based on the nodes coordinates and coresponding weights.
/// ## Inputs
/// - `ids`: global identifiers of the objects to partition
/// - `weights`: weights corsponding to a cost relative to the objects
/// - `coordinates`: the 2D coordinates of the objects to partition
///
/// ## Output
/// A Vec of couples `(usize, ProcessUniqueId)`
///
/// the first component of each couple is the id of an object and
/// the second component is the id of the partition to which that object was assigned
pub fn rcb(
    ids: Vec<usize>,
    weights: Vec<f64>,
    coordinates: Vec<Point2D>,
    n_iter: usize,
) -> Vec<(usize, ProcessUniqueId)> {
    rcb_impl(ids, weights, coordinates, n_iter, true)
}

fn rcb_impl(
    ids: Vec<usize>,
    weights: Vec<f64>,
    coordinates: Vec<Point2D>,
    n_iter: usize,
    x_axis: bool, // set if bissection is performed w.r.t. x or y axis
) -> Vec<(usize, ProcessUniqueId)> {
    if n_iter == 0 {
        let part_id = ProcessUniqueId::new();
        ids.into_iter().map(|id| (id, part_id)).collect()
    } else {
        let (mut ids, mut weights, mut coordinates) = axis_sort(ids, weights, coordinates, x_axis);
        let split_pos = half_weight_pos(weights.as_slice());

        let left_ids = ids.drain(0..split_pos).collect();
        let left_weights = weights.drain(0..split_pos).collect();
        let left_coordinates = coordinates.drain(0..split_pos).collect();

        let left_partition = rcb_impl(
            left_ids,
            left_weights,
            left_coordinates,
            n_iter - 1,
            !x_axis,
        );

        let right_partition = rcb_impl(ids, weights, coordinates, n_iter - 1, !x_axis);

        left_partition.into_iter().chain(right_partition).collect()
    }
}

// sort input vectors w.r.t. coordinates
// i.e. by increasing x or increasing y
fn axis_sort(
    ids: Vec<usize>,
    weights: Vec<f64>,
    coordinates: Vec<Point2D>,
    x_axis: bool,
) -> (Vec<usize>, Vec<f64>, Vec<Point2D>) {
    let sorted =
        itertools::multizip((ids, weights, coordinates)).sorted_by(|(_, _, p1), (_, _, p2)| {
            if x_axis {
                p1.x().partial_cmp(&p2.x()).unwrap_or(Ordering::Equal)
            } else {
                p1.y().partial_cmp(&p2.y()).unwrap_or(Ordering::Equal)
            }
        });

    let (ids, weights, coordinates) = unzip(sorted.into_iter());
    (ids.collect(), weights.collect(), coordinates.collect())
}

fn half_weight_pos(sorted_weights: &[f64]) -> usize {
    let mut half_weight = sorted_weights.iter().sum::<f64>() / 2.;

    let mut pos = 0;
    for w in sorted_weights {
        if half_weight > 0. {
            pos += 1;
        } else {
            break;
        }
        half_weight -= w;
    }

    pos
}

// quick and dirty equivalent of std::iter::unzip for itertools::multizip
fn unzip<A, B, C>(
    zipped: impl Iterator<Item = (A, B, C)> + Clone,
) -> (
    impl Iterator<Item = A>,
    impl Iterator<Item = B>,
    impl Iterator<Item = C>,
)
where
    A: Clone,
    B: Clone,
    C: Clone,
{
    (
        zipped.clone().map(|(a, _, _)| a.clone()),
        zipped.clone().map(|(_, b, _)| b),
        zipped.map(|(_, _, c)| c),
    )
}
