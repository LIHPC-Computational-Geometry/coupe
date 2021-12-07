use crate::geometry::Mbr;
use crate::geometry::PointND;

use itertools::Itertools as _;
use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use rayon::prelude::*;

use std::cmp;
use std::mem;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;

fn flatten_skip_take<T>(vs: &Vec<Vec<T>>, mut skip: usize, mut take: usize) -> Vec<T>
where
    T: Clone,
{
    let mut res = Vec::with_capacity(take);
    for v in vs {
        if v.len() <= skip {
            skip -= v.len();
            continue;
        }
        let end = usize::min(skip + take, v.len());
        res.extend_from_slice(&v[skip..end]);
        take -= end - skip;
        if take == 0 {
            break;
        }
        skip = 0;
    }
    res
}

#[derive(Clone, Debug)]
struct Item<'p, const D: usize> {
    point: PointND<D>,
    weight: f64,
    part: &'p AtomicUsize,
}

fn rcb_recurse<'p, const D: usize>(
    s: &rayon::Scope<'p>,
    items: crossbeam_channel::Receiver<Vec<Item<'p, D>>>,
    coord: usize,
    num_iter: usize,
    tolerance: f64,
    items_per_thread: usize,
) {
    use std::sync::atomic::AtomicBool;

    struct Locked {
        min: f64,
        max: f64,
        sum: f64,

        /// Weight on the left of the split target.  The weight on the right can
        /// be obtained through InitData's sum.
        left_part_weight: f64, // current iteration
        additional_left_weight: f64, // previous iterations

        /// The number of weights on the left of the split target.  The number
        /// on the right can be obtained through items's length.
        left_part_count: usize,

        /// The number of weights on the left of max and min.  Used to tell
        /// whether it is impossible to find a split target that satisfy the
        /// tolerance, and thus stop the iteration.
        count_left_max: usize,
        count_left_min: usize,

        /// Time-to-Live, initialized to the number of threads and decreased on
        /// every write.  To know when the next iteration can begin.
        ttl: usize,
    }

    /// Data shared by all threads
    struct ThreadContext {
        c: Condvar,
        data: Mutex<Locked>,

        /// whether to stop the computation
        run: AtomicBool,
        thread_count: AtomicUsize,
    }

    let collect_span = tracing::trace_span!("collect chunks from previous iter");
    let enter = collect_span.enter();

    let chunk_count = items.capacity().unwrap();
    let mut item_count = 0;
    let mut chunks = Vec::with_capacity(chunk_count);

    for _ in 0..chunk_count {
        let chunk = match items.recv() {
            Ok(chunk) => chunk,
            Err(_) => break,
        };
        item_count += chunk.len();
        chunks.push(chunk);
    }

    mem::drop(enter);

    if num_iter == 0 {
        s.spawn(move |_| {
            let part = crate::uid();

            let apply_span = tracing::debug_span!("apply_part_id", part_id = part);
            let _enter = apply_span.enter();

            for chunk in chunks {
                for item in chunk {
                    item.part.store(part, Ordering::Relaxed);
                }
            }
        });
        return;
    }

    let thread_count = usize::max(1, item_count / items_per_thread);
    let items_per_thread = (item_count + thread_count - 1) / thread_count;
    let ctx = Arc::new(ThreadContext {
        c: Condvar::new(),
        data: Mutex::new(Locked {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            left_part_weight: 0.0,
            additional_left_weight: 0.0,
            left_part_count: 0,
            count_left_max: item_count,
            count_left_min: 0,
            ttl: thread_count,
        }),
        thread_count: AtomicUsize::new(thread_count),
        run: AtomicBool::new(true),
    });

    let chunks = Arc::new(chunks);
    let wg = crossbeam_utils::sync::WaitGroup::new();
    let (left_tx, left_rx) = crossbeam_channel::bounded::<Vec<Item<'p, D>>>(thread_count);
    let (right_tx, right_rx) = crossbeam_channel::bounded::<Vec<Item<'p, D>>>(thread_count);

    for thread_chunk_start in (0..item_count).step_by(items_per_thread) {
        let ctx = Arc::clone(&ctx);
        let chunks = Arc::clone(&chunks);
        let wg = wg.clone();
        let left_tx = left_tx.clone();
        let left_rx = left_rx.clone();
        let right_tx = right_tx.clone();
        let right_rx = right_rx.clone();
        s.spawn(move |s| {
            let copy_span = tracing::trace_span!("copy items");
            let compute_span = tracing::trace_span!("compute");
            let first_sync_span = tracing::trace_span!("first sync");
            let sync_span = tracing::trace_span!("sync");
            let merge_span = tracing::trace_span!("merge");
            let partition_span = tracing::trace_span!("make partition");

            let enter = copy_span.enter();

            let mut thread_items =
                flatten_skip_take(&*chunks, thread_chunk_start, items_per_thread);

            mem::drop(enter);
            let enter = compute_span.enter();

            let partial_sum: f64 = thread_items.iter().map(|item| item.weight).sum();
            let (partial_min, partial_max) = thread_items
                .iter()
                .map(|item| item.point[coord])
                .minmax()
                .into_option()
                .unwrap();

            mem::drop(enter);
            let enter = first_sync_span.enter();

            let mut min: f64;
            let mut max: f64;
            {
                let mut d = ctx.data.lock().unwrap();
                if partial_min < d.min {
                    d.min = partial_min;
                }
                if d.max < partial_max {
                    d.max = partial_max;
                }
                d.sum += partial_sum;
            }
            wg.wait();
            {
                let d = ctx.data.lock().unwrap();
                min = d.min;
                max = d.max;
            }

            mem::drop(enter);

            let initial_min = min;
            let initial_max = max;

            let mut lead = false;
            let mut item_view = &mut *thread_items;
            let mut additional_partial_left_count = 0;
            let mut median_idx = 0;
            while ctx.run.load(Ordering::SeqCst) {
                let enter = compute_span.enter();

                let split_target = (min + max) / 2.0;

                // Split thread items into two, separated by the split target.

                let partial_left_count = item_view
                    .iter()
                    .filter(|item| item.point[coord] < split_target)
                    .count();
                let (partial_left, partial_right) = if partial_left_count == item_view.len() {
                    let empty: &mut [Item<'p, D>] = &mut [];
                    (item_view, empty)
                } else {
                    let (partial_left, _, partial_right) =
                        item_view.select_nth_unstable_by(partial_left_count, |item1, item2| {
                            f64::partial_cmp(&item1.point[coord], &item2.point[coord]).unwrap()
                        });
                    (partial_left, partial_right)
                };
                let partial_left_weight: f64 = partial_left.iter().map(|item| item.weight).sum();
                let partial_left_count = partial_left_count + additional_partial_left_count;

                // Update shared state

                mem::drop(enter);
                let enter = sync_span.enter();

                let mut data = ctx.data.lock().unwrap();

                mem::drop(enter);
                let enter = merge_span.enter();

                if data.max - data.min != max - min {
                    // This thread is lagging behind: discard work, advance to the
                    // correct split target and restart.
                    min = data.min;
                    max = data.max;
                    if split_target <= data.min {
                        median_idx = 0;
                        item_view = partial_right;
                        additional_partial_left_count = partial_left_count;
                    } else {
                        median_idx = partial_left.len();
                        item_view = partial_left;
                    }
                    if item_view.is_empty() {
                        ctx.thread_count.fetch_sub(1, Ordering::Relaxed);
                        data.ttl = data.ttl.checked_sub(1).unwrap();
                        break;
                    }
                    continue;
                }

                data.ttl -= 1;
                data.left_part_count += partial_left_count;
                data.left_part_weight += partial_left_weight;

                let weight_left = data.additional_left_weight + data.left_part_weight;
                let remaining_weight = data.sum - weight_left;

                if remaining_weight < weight_left {
                    // Weight of the items on the left is already dominating, no
                    // need to compute any further.
                    median_idx = partial_left.len();
                    item_view = partial_left;
                    max = split_target;
                    data.max = split_target;
                    if data.ttl == 0 {
                        data.count_left_max = data.left_part_count;
                    }
                } else if data.ttl == 0 {
                    // This thread is the last one to update the state, prepare for
                    // the next iteration.
                    median_idx = 0;
                    item_view = partial_right;
                    additional_partial_left_count = partial_left_count;
                    min = split_target;
                    data.min = split_target;
                    data.count_left_min = data.left_part_count;
                    data.additional_left_weight = weight_left;
                } else {
                    mem::drop(enter);
                    let enter = sync_span.enter();

                    data = ctx
                        .c
                        .wait_while(data, |data| data.max == max && data.min == min)
                        .unwrap();

                    mem::drop(enter);
                    let _enter = merge_span.enter();

                    min = data.min;
                    max = data.max;
                    if split_target <= data.min {
                        median_idx = 0;
                        item_view = partial_right;
                        additional_partial_left_count = partial_left_count;
                    } else {
                        median_idx = partial_left.len();
                        item_view = partial_left;
                    }
                    if item_view.is_empty() {
                        ctx.thread_count.fetch_sub(1, Ordering::Relaxed);
                        data.ttl = data.ttl.checked_sub(1).unwrap();
                        break;
                    }
                    continue;
                }

                if f64::abs(weight_left - remaining_weight) < tolerance * data.sum
                    || max - min < tolerance * (initial_max - initial_min)
                    || data.count_left_min == data.count_left_max
                {
                    lead = ctx
                        .run
                        .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok(); // is always ok for now, since we're inside a lock
                }

                data.left_part_count = 0;
                data.left_part_weight = 0.0;
                data.ttl = ctx.thread_count.load(Ordering::Relaxed);

                ctx.c.notify_all();

                if item_view.is_empty() {
                    ctx.thread_count.fetch_sub(1, Ordering::Relaxed);
                    break;
                }
            }

            if ctx.thread_count.load(Ordering::Relaxed) == 0 && !lead {
                // run can still be true if all threads exited early
                lead = ctx
                    .run
                    .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok();
            }

            let enter = partition_span.enter();

            let median_idx = additional_partial_left_count + median_idx;
            let right = thread_items.split_off(median_idx);
            left_tx.send(thread_items).unwrap();
            right_tx.send(right).unwrap();

            mem::drop(enter);

            if lead {
                let next_coord = (coord + 1) % D;

                rcb_recurse(
                    s,
                    left_rx,
                    next_coord,
                    num_iter - 1,
                    tolerance,
                    items_per_thread,
                );
                rcb_recurse(
                    s,
                    right_rx,
                    next_coord,
                    num_iter - 1,
                    tolerance,
                    items_per_thread,
                );
            }
        });
    }
}

pub fn rcb<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    iter_count: usize,
) -> Vec<usize> {
    let mut partition = vec![0; points.len()];

    let items: Vec<_> = points
        .par_iter()
        .zip(weights)
        .zip(unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(&mut partition) })
        .map(|((&point, &weight), part)| Item {
            point,
            weight,
            part,
        })
        .collect();

    rayon::in_place_scope(|s| {
        let thread_count = rayon::current_num_threads();
        let items_per_thread = (items.len() + thread_count - 1) / thread_count;

        let (item_tx, item_rx) = crossbeam_channel::bounded(1);
        item_tx.send(items).unwrap();

        rcb_recurse(s, item_rx, 0, iter_count, 0.05, items_per_thread);
    });

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
pub fn rib<const D: usize>(points: &[PointND<D>], weights: &[f64], n_iter: usize) -> Vec<usize>
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
        let mut loads: HashMap<usize, f64> = HashMap::new();
        let mut sizes: HashMap<usize, usize> = HashMap::new();
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
