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

use async_lock::Mutex;
use async_lock::MutexGuard;
use std::cmp;
use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Default)]
struct Condvar {
    event: event_listener::Event,
}

impl Condvar {
    pub async fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
        let listener = self.event.listen();
        let lock = MutexGuard::source(&guard);
        mem::drop(guard);
        listener.await;
        lock.lock().await
    }

    pub fn notify_all(&self) {
        self.event.notify(usize::MAX);
    }
}

#[derive(Clone, Debug)]
struct Item<'p, const D: usize> {
    point: PointND<D>,
    weight: f64,
    part: &'p AtomicUsize,
}

#[derive(Debug, Default)]
struct IterationData {
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
    /// every write on min, max and sum, to know when they are initialized.
    ttl_min_max_sum: usize,

    ttl: usize,
}

impl IterationData {
    pub fn new(thread_count: usize) -> Self {
        Self {
            ttl_min_max_sum: thread_count,
            ..IterationData::default()
        }
    }
}

/// Data shared by all threads
struct IterationState {
    c: Condvar,
    data: Mutex<IterationData>,

    /// whether to stop the computation
    run: AtomicBool,
    thread_count: AtomicUsize,
}

impl IterationState {
    fn new(thread_count: usize) -> Self {
        Self {
            c: Condvar::default(),
            data: Mutex::new(IterationData::new(thread_count)),
            run: AtomicBool::new(true),
            thread_count: AtomicUsize::new(thread_count),
        }
    }
}

#[tracing::instrument(skip(items))]
fn apply_part_id<const D: usize>(items: &[Item<'_, D>], part_id: usize) {
    for item in items {
        item.part.store(part_id, Ordering::Relaxed);
    }
}

#[tracing::instrument(skip(iter_ctxs))]
async fn cancel_iterations<const D: usize>(
    iter_ctxs: &[IterationState],
    iter_count: usize,
    iter_id: usize,
) {
    let cancel_task = async {
        let ctx = &iter_ctxs[iter_id];
        let mut d = ctx.data.lock().await;
        d.ttl_min_max_sum = d.ttl_min_max_sum.checked_sub(1).unwrap();
        // We won't participate in the iteration.
        ctx.thread_count.fetch_sub(1, Ordering::Relaxed);
        ctx.c.notify_all();
    };
    let left_task = rcb_iter::<D>(iter_ctxs, &mut [], iter_count - 1, iter_id * 2 + 1, 0, 0.0);
    let right_task = rcb_iter::<D>(iter_ctxs, &mut [], iter_count - 1, iter_id * 2 + 2, 0, 0.0);

    let left_right_tasks = futures_lite::future::zip(left_task, right_task);
    futures_lite::future::zip(cancel_task, left_right_tasks).await;
}

#[tracing::instrument(skip(ctx, items), ret)]
async fn compute_min_max<const D: usize>(
    ctx: &IterationState,
    items: &[Item<'_, D>],
    coord: usize,
) -> (f64, f64) {
    let min_max_span = tracing::info_span!("compute");
    let enter = min_max_span.enter();

    let partial_sum: f64 = items.iter().map(|item| item.weight).sum();
    let (partial_min, partial_max) = items
        .iter()
        .map(|item| item.point[coord])
        .minmax()
        .into_option()
        .unwrap();

    mem::drop(enter);

    let mut d = ctx.data.lock().await;
    if partial_min < d.min {
        d.min = partial_min;
    }
    if d.max < partial_max {
        d.max = partial_max;
    }
    d.sum += partial_sum;
    d.ttl_min_max_sum = d.ttl_min_max_sum.checked_sub(1).unwrap();
    if d.ttl_min_max_sum == 0 {
        ctx.c.notify_all();
    }
    while d.ttl_min_max_sum != 0 {
        d = ctx.c.wait(d).await;
    }
    if d.ttl == 0 {
        // Only set these values once.
        d.ttl = ctx.thread_count.load(Ordering::Relaxed);
        d.count_left_max = items.len();
    }

    (d.min, d.max)
}

struct SplitResult<'a, 'p, const D: usize> {
    left: &'a mut [Item<'p, D>],
    right: &'a mut [Item<'p, D>],
    left_weight: f64,
}

#[tracing::instrument(skip(items))]
fn try_split_items<'a, 'p, const D: usize>(
    items: &'a mut [Item<'p, D>],
    coord: usize,
    split_target: f64,
) -> SplitResult<'a, 'p, D> {
    let left_count = items
        .iter()
        .filter(|item| item.point[coord] < split_target)
        .count();

    let (left, right) = if left_count == items.len() {
        items.split_at_mut(items.len())
    } else {
        let (left, _, _right_minus_one) = items
            .select_nth_unstable_by(left_count, |item1, item2| {
                f64::partial_cmp(&item1.point[coord], &item2.point[coord]).unwrap()
            });
        let left_len = left.len();
        items.split_at_mut(left_len)
    };

    let left_weight: f64 = left.iter().map(|item| item.weight).sum();

    SplitResult {
        left,
        right,
        left_weight,
    }
}

#[derive(Copy, Clone, Debug)]
enum Direction {
    Left,
    Right,
}

#[derive(Debug)]
struct SyncItemResult {
    goto: Direction,
    left_weight: f64,
    right_weight: f64,
}

#[tracing::instrument(skip(ctx), ret)]
async fn sync_item_split<'d, 'p>(
    ctx: &IterationState,
    mut data: MutexGuard<'d, IterationData>,
    min: f64,
    max: f64,
    thread_left_count: usize,
    thread_left_weight: f64,
) -> (MutexGuard<'d, IterationData>, Option<SyncItemResult>) {
    if max != data.max || min != data.min {
        // This thread is lagging behind: discard work, advance to the
        // correct split target and restart.
        return (data, None);
    }

    data.ttl = data.ttl.checked_sub(1).unwrap();
    data.left_part_count += thread_left_count;
    data.left_part_weight += thread_left_weight;

    let mut left_weight = data.additional_left_weight + data.left_part_weight;
    let mut remaining_weight = data.sum - left_weight;

    let goto = if remaining_weight < left_weight {
        // Weight of the items on the left is already dominating, no
        // need to compute any further.
        Direction::Left
    } else if data.ttl == 0 {
        // This thread is the last one to update the state, prepare for
        // the next iteration.
        Direction::Right
    } else {
        while max == data.max && min == data.min && data.ttl != 0 {
            data = ctx.c.wait(data).await;
        }
        if data.ttl == 0 {
            // Some threads have left and the next split_target has not
            // been chosen. Since no thread has hit the branch where
            // the remaining_weight is lower that left_weight, it means
            // left_weight < right_weight.
            // Don't forget to also update left_weight and remaining_weight!
            left_weight = data.additional_left_weight + data.left_part_weight;
            remaining_weight = data.sum - left_weight;
            Direction::Right
        } else {
            // All other threads have finished the iteration and the
            // next one has started, prepare for it.
            return (data, None);
        }
    };

    let res = SyncItemResult {
        goto,
        left_weight,
        right_weight: remaining_weight,
    };
    (data, Some(res))
}

#[tracing::instrument(skip(ctx, items), ret)]
async fn item_split_idx<'p, const D: usize>(
    ctx: &IterationState,
    items: &mut [Item<'p, D>],
    coord: usize,
    tolerance: f64,
) -> usize {
    let (mut min, mut max) = compute_min_max(ctx, items, coord).await;

    let min_search_space = (max - min) * tolerance;
    let mut item_view = &mut *items;
    let mut median_idx = 0;
    let mut cached_thread_left_count = 0;
    let mut cached_thread_left_weight = 0.0;

    while ctx.run.load(Ordering::Relaxed) {
        let split_target = (min + max) / 2.0;
        let split = try_split_items(item_view, coord, split_target);
        let thread_left_count = split.left.len() + cached_thread_left_count;
        let thread_left_weight = split.left_weight + cached_thread_left_weight;

        let data = ctx.data.lock().await;
        let (mut data, update_global_state) =
            sync_item_split(ctx, data, min, max, thread_left_count, thread_left_weight).await;

        let goto = match &update_global_state {
            Some(SyncItemResult { goto, .. }) => *goto,
            None => {
                min = data.min;
                max = data.max;
                if split_target <= data.min {
                    Direction::Right
                } else {
                    Direction::Left
                }
            }
        };

        match goto {
            Direction::Left => {
                median_idx = split.left.len();
                item_view = split.left;
            }
            Direction::Right => {
                median_idx = 0;
                item_view = split.right;
                cached_thread_left_count = thread_left_count;
                cached_thread_left_weight = thread_left_weight;
            }
        }
        if item_view.is_empty() {
            ctx.thread_count.fetch_sub(1, Ordering::Relaxed);
        }

        match update_global_state {
            Some(sync) => {
                match goto {
                    Direction::Left => {
                        max = split_target;
                        data.max = split_target;
                        if data.ttl == 0 {
                            // Only update number of weights below data.max if all
                            // threads have contributed to data.left_part_count.
                            data.count_left_max = data.left_part_count;
                        }
                    }
                    Direction::Right => {
                        min = split_target;
                        data.min = split_target;
                        data.count_left_min = data.left_part_count;
                        data.additional_left_weight = sync.left_weight;
                    }
                }
                let respect_tolerance =
                    f64::abs(sync.left_weight - sync.right_weight) < tolerance * data.sum;
                let small_search_space = max - min < min_search_space;
                let empty_search_space = data.count_left_min == data.count_left_max;
                if respect_tolerance || small_search_space || empty_search_space {
                    tracing::info!(
                        respect_tolerance,
                        small_search_space,
                        empty_search_space,
                        "stopped iteration",
                    );
                    ctx.run.store(false, Ordering::Relaxed);
                }

                data.left_part_count = 0;
                data.left_part_weight = 0.0;
                data.ttl = ctx.thread_count.load(Ordering::Relaxed);
                ctx.c.notify_all();
                if item_view.is_empty() {
                    break;
                }
            }
            None => {
                if item_view.is_empty() {
                    data.ttl = data.ttl.checked_sub(1).unwrap();
                    ctx.c.notify_all();
                    break;
                }
            }
        }
    }

    median_idx + cached_thread_left_count
}

fn rcb_iter<'p, const D: usize>(
    iter_ctxs: &'p [IterationState],
    items: &'p mut [Item<'p, D>],
    iter_count: usize,
    iter_id: usize,
    coord: usize,
    tolerance: f64,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'p>> {
    use tracing::Instrument as _;

    let fut = async move {
        if iter_count == 0 {
            // No need to split, all thoses items are in the same part.
            apply_part_id(items, iter_id);
            return;
        }
        if items.is_empty() {
            // No items left in this thread, remove it from global state.
            cancel_iterations::<D>(iter_ctxs, iter_count, iter_id).await;
            return;
        }

        let ctx = &iter_ctxs[iter_id];
        let split_idx = item_split_idx(ctx, &mut *items, coord, tolerance).await;
        let (left, right) = items.split_at_mut(split_idx);

        tracing::info!("cut result: left={}, right={}", left.len(), right.len());

        let left_task = rcb_iter(
            iter_ctxs,
            left,
            iter_count - 1,
            iter_id * 2 + 1,
            (coord + 1) % D,
            tolerance,
        );
        let right_task = rcb_iter(
            iter_ctxs,
            right,
            iter_count - 1,
            iter_id * 2 + 2,
            (coord + 1) % D,
            tolerance,
        );
        futures_lite::future::zip(left_task, right_task).await;
    };

    let span = tracing::info_span!("rcb_iter", iter_count, iter_id);
    Box::pin(fut.instrument(span))
}

fn rcb_thread<const D: usize>(
    iter_ctxs: &[IterationState],
    items: &[Item<'_, D>],
    iter_count: usize,
    tolerance: f64,
) {
    let copy_span = tracing::info_span!("copy items");
    let enter = copy_span.enter();

    let mut items = items.to_vec();

    mem::drop(enter);

    let task = rcb_iter(iter_ctxs, &mut items, iter_count, 0, 0, tolerance);
    futures_lite::future::block_on(task);
}

pub fn rcb<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    iter_count: usize,
) -> Vec<usize> {
    let init_span = tracing::info_span!("convert input and make initial data structures");
    let enter = init_span.enter();

    let mut partition = vec![0; points.len()];

    let mut items: Vec<_> = points
        .par_iter()
        .zip(weights)
        .zip(unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(&mut partition) })
        .map(|((&point, &weight), part)| Item {
            point,
            weight,
            part,
        })
        .collect();

    let iteration_ctxs: Vec<_> = (0..usize::pow(2, iter_count as u32 + 1) - 1)
        .map(|_| IterationState::new(rayon::current_num_threads()))
        .collect();

    mem::drop(enter);

    rayon::in_place_scope(|s| {
        let thread_count = rayon::current_num_threads();
        let items_per_thread = (items.len() + thread_count - 1) / thread_count;

        for chunk in items.chunks_mut(items_per_thread) {
            let iteration_ctxs = &iteration_ctxs;
            s.spawn(move |_| rcb_thread(iteration_ctxs, chunk, iter_count, 0.05));
        }
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

        let partition = rayon::ThreadPoolBuilder::new()
            .num_threads(1) // make the test deterministic
            .build()
            .unwrap()
            .install(|| rcb(&points, &weights, 2));

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

    //#[test] // Disabled by default because of its need for a random source.
    fn _test_rcb_rand() {
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
            println!("{part:?} -> {size}:{load:.1}");
        }
    }
}
