use crate::geometry::Mbr;
use crate::geometry::PointND;

use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp;
use std::sync;

struct Item<'p, const D: usize> {
    point: PointND<D>,
    weight: f64,
    part: &'p mut ProcessUniqueId,
}

fn weighted_median<'p, const D: usize>(
    mut items: Vec<Item<'p, D>>,
    coord: usize,
    tolerance: f64,
    num_avail_threads: usize,
) -> (Vec<Item<'p, D>>, Vec<Item<'p, D>>) {
    let split_target = crossbeam_utils::thread::scope(|s| {
        let num_items = items.len();
        let chunk_size = usize::max((num_items + num_avail_threads - 1) / num_avail_threads, 64);
        let num_threads = (num_items + chunk_size - 1) / chunk_size;
        #[cfg(debug_assertions)]
        println!(
            "avail_threads={},num_threads={},num_items={},chunk_size={}",
            num_avail_threads, num_threads, num_items, chunk_size,
        );

        let (sender, receiver) = sync::mpsc::sync_channel(num_threads);
        let split_target: sync::Arc<sync::RwLock<Option<f64>>> =
            sync::Arc::new(sync::RwLock::new(None));
        let barrier = sync::Arc::new(sync::Barrier::new(num_threads + 1));

        #[cfg(debug_assertions)]
        let mut i = 0;
        for item_chunk in items.chunks_mut(chunk_size) {
            let split_target = split_target.clone();
            let sender = sender.clone();
            let barrier = barrier.clone();
            s.spawn(move |_| {
                #[cfg(debug_assertions)]
                println!("compute {}: spawn", i);
                struct ThreadItem {
                    point_coord: f64,
                    weight: f64,
                }
                let thread_items: Vec<_> = item_chunk
                    .into_iter()
                    .map(|item| ThreadItem {
                        point_coord: item.point[coord],
                        weight: item.weight,
                    })
                    .collect();
                let sum: f64 = thread_items.iter().map(|item| item.weight).sum();

                use itertools::Itertools as _;
                let (min, max) = thread_items
                    .iter()
                    .map(|item| item.point_coord)
                    .minmax()
                    .into_option()
                    .unwrap();

                #[cfg(debug_assertions)]
                println!("compute {}: send sum/min/max", i);
                sender.send((sum, min, max)).unwrap();
                #[cfg(debug_assertions)]
                println!("compute {}: waiting for barrier", i);
                barrier.wait();
                #[cfg(debug_assertions)]
                println!("compute {}: woke up", i);

                loop {
                    let split_target = match *split_target.try_read().unwrap() {
                        Some(val) => val,
                        None => break,
                    };
                    let (weight_left, weight_right) = thread_items
                        .iter()
                        .map(|item| {
                            if item.point_coord < split_target {
                                (item.weight, 0.0)
                            } else {
                                (0.0, item.weight)
                            }
                        })
                        .fold((0.0, 0.0), |(wl0, wr0), (wl1, wr1)| (wl0 + wl1, wr0 + wr1));
                    #[cfg(debug_assertions)]
                    println!("compute {}: send weight_left/right", i);
                    sender.send((weight_left, weight_right, 0.0)).unwrap();
                    #[cfg(debug_assertions)]
                    println!("compute {}: waiting for barrier", i);
                    barrier.wait();
                    #[cfg(debug_assertions)]
                    println!("compute {}: woke up", i);
                }
            });
            #[cfg(debug_assertions)]
            {
                i += 1;
            }
        }
        #[cfg(debug_assertions)]
        assert_eq!(i, num_threads,);

        let mut sum = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for times_to_recv in 0..num_threads {
            #[cfg(debug_assertions)]
            println!("main: receiving sum/min/max ({})", times_to_recv);
            let (partial_sum, partial_min, partial_max) = receiver.recv().unwrap();
            sum += partial_sum;
            if partial_min < min {
                min = partial_min;
            }
            if max < partial_max {
                max = partial_max;
            }
        }
        let max_imbalance = tolerance * sum;
        let mut prev_weight_left = 0.0;
        let mut split_target_val = (max + min) / 2.0;
        #[cfg(debug_assertions)]
        println!("main: writing split_target");
        *split_target.try_write().unwrap() = Some(split_target_val);
        #[cfg(debug_assertions)]
        println!("main: waiting for barrier");
        barrier.wait();
        #[cfg(debug_assertions)]
        println!("main: woke up");

        loop {
            let mut weight_left = 0.0;
            let mut weight_right = 0.0;
            for _times_to_recv in 0..num_threads {
                #[cfg(debug_assertions)]
                println!("main: receiving weight_left/right");
                let (partial_weight_left, partial_weight_right, _) = receiver.recv().unwrap();
                weight_left += partial_weight_left;
                weight_right += partial_weight_right;
            }

            let imbalance = f64::abs(weight_left - weight_right);
            if imbalance < max_imbalance || prev_weight_left == weight_left {
                #[cfg(debug_assertions)]
                println!("main: exiting split_target");
                *split_target.try_write().unwrap() = None; // shutdown other threads
                #[cfg(debug_assertions)]
                println!("main: waiting for barrier");
                barrier.wait();
                #[cfg(debug_assertions)]
                println!("main: woke up");
                break;
            }

            if weight_left < weight_right {
                min = split_target_val;
            } else {
                max = split_target_val;
            }
            split_target_val = (max + min) / 2.0;
            #[cfg(debug_assertions)]
            println!("main: writing split_target");
            *split_target.try_write().unwrap() = Some(split_target_val);
            #[cfg(debug_assertions)]
            println!("main: waiting for barrier");
            barrier.wait();
            #[cfg(debug_assertions)]
            println!("main: woke up");

            prev_weight_left = weight_left;
        }
        split_target_val
    });

    let split_target = split_target.unwrap();
    items
        .into_par_iter()
        .partition(|item| item.point[coord] < split_target)
}

fn rcb_recurse<'p, const D: usize>(
    s: &crossbeam_utils::thread::Scope<'p>,
    items: Vec<Item<'p, D>>,
    num_iter: usize,
    coord: usize,
    tolerance: f64,
    num_avail_threads: usize,
) {
    if num_iter == 0 {
        let part = ProcessUniqueId::new();
        items.into_iter().for_each(|item| *item.part = part);
        return;
    }
    let (left, right) = weighted_median(items, coord, tolerance, num_avail_threads);

    let next_coord = (coord + 1) % D;
    let next_num_avail_threads = usize::max(num_avail_threads / 2, 1);
    s.spawn(move |s| {
        rcb_recurse(
            s,
            left,
            num_iter - 1,
            next_coord,
            tolerance,
            next_num_avail_threads,
        )
    });
    s.spawn(move |s| {
        rcb_recurse(
            s,
            right,
            num_iter - 1,
            next_coord,
            tolerance,
            next_num_avail_threads,
        )
    });
}

pub fn rcb<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    n_iter: usize,
) -> Vec<ProcessUniqueId> {
    let dummy_id = ProcessUniqueId::new();
    let mut partition = vec![dummy_id; points.len()];

    crossbeam_utils::thread::scope(|s| {
        let items = points
            .par_iter()
            .zip(weights)
            .zip(&mut partition)
            .map(|((&point, &weight), part)| Item {
                point,
                weight,
                part,
            })
            .collect();
        rcb_recurse(s, items, n_iter, 0, 0.05, rayon::current_num_threads());
    })
    .unwrap();

    partition
}

// pub because it is also useful for multijagged and required for benchmarks
pub fn axis_sort<const D: usize>(
    points: &[PointND<D>],
    permutation: &mut [usize],
    current_coord: usize,
) {
    permutation.par_sort_by(|i1, i2| {
        if points[*i1][current_coord] < points[*i2][current_coord] {
            cmp::Ordering::Less
        } else {
            cmp::Ordering::Greater
        }
    })
}

/// # Recursive Inertia Bisection algorithm
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
///
/// The main difference with the RCB algorithm is that, in RCB, points are split
/// with a separator which is parallel to either the x axis or the y axis. With RIB,
/// The global shape of the data is first considered and the separator is computed to
/// be parallel to the inertia axis of the global shape, which aims to lead to better shaped
/// partitions.
pub fn rib<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    n_iter: usize,
) -> Vec<ProcessUniqueId>
where
    Const<D>: DimSub<Const<1>>,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
{
    let mbr = Mbr::from_points(points);

    let points = points
        .par_iter()
        .map(|p| mbr.mbr_to_aabb(p))
        .collect::<Vec<_>>();

    // When the rotation is done, we just apply RCB
    rcb(&points, weights, n_iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Point2D;

    fn gen_point_sample() -> Vec<Point2D> {
        vec![
            Point2D::from([4., 6.]),
            Point2D::from([9., 5.]),
            Point2D::from([-1.2, 7.]),
            Point2D::from([0., 0.]),
            Point2D::from([3., 9.]),
            Point2D::from([-4., 3.]),
            Point2D::from([1., 2.]),
        ]
    }

    #[test]
    fn test_axis_sort_x() {
        let points = gen_point_sample();
        let mut permutation = (0..points.len()).collect::<Vec<usize>>();

        axis_sort(&points, &mut permutation, 0);

        assert_eq!(permutation, vec![5, 2, 3, 6, 4, 0, 1]);
    }

    #[test]
    fn test_axis_sort_y() {
        let points = gen_point_sample();
        let mut permutation = (0..points.len()).collect::<Vec<usize>>();

        axis_sort(&points, &mut permutation, 1);

        assert_eq!(permutation, vec![3, 6, 5, 1, 0, 2, 4]);
    }

    #[test]
    fn test_rcb_basic() {
        let weights = vec![1.; 8];
        let points = vec![
            Point2D::from([-1.3, 6.]),
            Point2D::from([2., -4.]),
            Point2D::from([1., 1.]),
            Point2D::from([-3., -2.5]),
            Point2D::from([-1.3, -0.3]),
            Point2D::from([2., 1.]),
            Point2D::from([-3., 1.]),
            Point2D::from([1.3, -2.]),
        ];

        let partition = rcb(&points, &weights, 2);

        assert_eq!(partition[0], partition[6]);
        assert_eq!(partition[1], partition[7]);
        assert_eq!(partition[2], partition[5]);
        assert_eq!(partition[3], partition[4]);

        let (p_id1, p_id2, p_id3, p_id4) = (partition[0], partition[1], partition[2], partition[3]);

        let p1 = partition.iter().filter(|p_id| **p_id == p_id1);
        let p2 = partition.iter().filter(|p_id| **p_id == p_id2);
        let p3 = partition.iter().filter(|p_id| **p_id == p_id3);
        let p4 = partition.iter().filter(|p_id| **p_id == p_id4);

        assert_eq!(p1.count(), 2);
        assert_eq!(p2.count(), 2);
        assert_eq!(p3.count(), 2);
        assert_eq!(p4.count(), 2);
    }

    #[test]
    fn test_rcb_rand() {
        use std::collections::HashMap;

        let points: Vec<Point2D> = (0..40000)
            .map(|_| Point2D::from([rand::random(), rand::random()]))
            .collect();
        let weights: Vec<f64> = (0..points.len()).map(|_| rand::random()).collect();
        let partition = rcb(&points, &weights, 3);
        let mut loads: HashMap<ProcessUniqueId, f64> = HashMap::new();
        let mut sizes: HashMap<ProcessUniqueId, usize> = HashMap::new();
        for (weight_id, part) in partition.iter().enumerate() {
            let weight = weights[weight_id];
            *loads.entry(*part).or_default() += weight;
            *sizes.entry(*part).or_default() += 1;
        }
        for ((part, load), size) in loads.iter().zip(sizes.values()) {
            println!("{:?} -> {}:{}", part, size, load);
        }
    }
}
