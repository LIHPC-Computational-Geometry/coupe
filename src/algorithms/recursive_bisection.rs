use super::Error;
use crate::geometry::Mbr;
use crate::geometry::PointND;
use async_lock::Mutex;
use async_lock::MutexGuard;
use itertools::Itertools as _;
use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use nalgebra::ToTypenum;
use rayon::prelude::*;
use std::cmp;
use std::future::Future;
use std::iter::Sum;
use std::mem;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::pin::Pin;
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
struct Item<'p, const D: usize, W> {
    point: PointND<D>,
    weight: W,
    part: &'p AtomicUsize,
}

#[derive(Debug, Default)]
struct IterationData<W> {
    min: f64,
    max: f64,
    sum: W,

    /// Weight on the left of the split target.  The weight on the right can
    /// be obtained through InitData's sum.
    left_part_weight: W, // current iteration
    additional_left_weight: W, // previous iterations

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

impl<W> IterationData<W>
where
    W: Default,
{
    pub fn new(thread_count: usize) -> Self {
        Self {
            ttl_min_max_sum: thread_count,
            ..IterationData::default()
        }
    }
}

/// Data shared by all threads
struct IterationState<W> {
    c: Condvar,
    data: Mutex<IterationData<W>>,

    /// whether to stop the computation
    run: AtomicBool,
    thread_count: AtomicUsize,
}

impl<W> IterationState<W>
where
    W: Default,
{
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
fn apply_part_id<const D: usize, W>(items: &[Item<'_, D, W>], part_id: usize) {
    for item in items {
        item.part.store(part_id, Ordering::Relaxed);
    }
}

#[tracing::instrument(skip(iter_ctxs))]
async fn cancel_iterations<W>(iter_ctxs: &[IterationState<W>], iter_count: usize, iter_id: usize) {
    // We start with iteration `iter_id` and visit all its transitive children,
    // breadth-first style.
    // `iter_ctxs` is ordered in such manner that the children of iteration N
    // are iterations 2N+1 and 2N+2.  Thus, its 2nd-level children are
    // iterations 4N+3, 4N+4, 4N+5 and 4N+6.  In general, its p-level children
    // are iterations [2^p*N + 2^p-1; 2^p*N + 2^(p+1)-1[.
    let mut i = 0; // offset value, ranging from 2^1-1 to 2^((iter_count-1)+1)-1
    let mut iter_level = 1; // 2^p
    for _ in 0..iter_count {
        for _ in 0..iter_level {
            let ctx = &iter_ctxs[iter_id * iter_level + i];
            let mut d = ctx.data.lock().await;
            d.ttl_min_max_sum = d.ttl_min_max_sum.checked_sub(1).unwrap();
            // We won't participate in the iteration.
            ctx.thread_count.fetch_sub(1, Ordering::Relaxed);
            ctx.c.notify_all();
            i += 1;
        }
        iter_level *= 2;
    }
}

/// # Panics
///
/// This function panics if `items` is empty.
#[tracing::instrument(skip(ctx, items))]
async fn compute_min_max<const D: usize, W>(
    ctx: &IterationState<W>,
    items: &[Item<'_, D, W>],
    coord: usize,
) -> (f64, f64)
where
    W: Copy + AddAssign + Sum + std::fmt::Debug,
{
    let min_max_span = tracing::info_span!("compute");
    let enter = min_max_span.enter();

    let partial_sum: W = items.iter().map(|item| item.weight).sum();
    let (partial_min, partial_max) = items
        .iter()
        .map(|item| item.point[coord])
        .minmax()
        .into_option()
        .unwrap(); // Won't panic because items has at least one element.

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

    tracing::info!(
        "compute_min_max_sum: min={:?}/{:?}, max={:?}/{:?}, sum={:?}/{:?}",
        partial_min,
        d.min,
        partial_max,
        d.max,
        partial_sum,
        d.sum,
    );

    (d.min, d.max)
}

struct SplitResult<'a, 'p, const D: usize, W> {
    left: &'a mut [Item<'p, D, W>],
    right: &'a mut [Item<'p, D, W>],
    left_weight: W,
}

#[tracing::instrument(skip(items))]
fn try_split_items<'a, 'p, const D: usize, W>(
    items: &'a mut [Item<'p, D, W>],
    coord: usize,
    split_target: f64,
) -> SplitResult<'a, 'p, D, W>
where
    W: Copy + Sum,
{
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

    let left_weight: W = left.iter().map(|item| item.weight).sum();

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
struct SyncItemResult<W> {
    goto: Direction,
    left_weight: W,
    right_weight: W,
}

#[tracing::instrument(skip(ctx), ret)]
async fn sync_item_split<'d, 'p, W>(
    ctx: &IterationState<W>,
    mut data: MutexGuard<'d, IterationData<W>>,
    min: f64,
    max: f64,
    thread_left_count: usize,
    thread_left_weight: W,
) -> (MutexGuard<'d, IterationData<W>>, Option<SyncItemResult<W>>)
where
    W: Copy + std::fmt::Debug,
    W: Add<Output = W> + AddAssign + Sub<Output = W> + PartialOrd,
{
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

/// # Panics
///
/// This function panics if `items` is empty.
#[tracing::instrument(skip(ctx, items), ret)]
async fn item_split_idx<'p, const D: usize, W>(
    ctx: &IterationState<W>,
    items: &mut [Item<'p, D, W>],
    coord: usize,
    tolerance: f64,
) -> usize
where
    W: Copy + std::fmt::Debug + Default,
    W: Add<Output = W> + AddAssign + Sub<Output = W> + Sum + PartialOrd,
    W: num::ToPrimitive,
{
    let (mut min, mut max) = compute_min_max(ctx, items, coord).await;

    let min_search_space = (max - min) * tolerance;
    let mut item_view = &mut *items;
    let mut median_idx = 0;
    let mut cached_thread_left_count = 0;
    let mut cached_thread_left_weight = W::default();

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
                let respect_tolerance = {
                    let imbalance = (sync.left_weight - sync.right_weight)
                        .to_f64()
                        .unwrap()
                        .abs();
                    let max_imbalance = tolerance * data.sum.to_f64().unwrap();
                    imbalance < max_imbalance
                };
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
                data.left_part_weight = W::default();
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

fn rcb_iter<'p, const D: usize, W>(
    iter_ctxs: &'p [IterationState<W>],
    items: &'p mut [Item<'p, D, W>],
    iter_count: usize,
    iter_id: usize,
    coord: usize,
    tolerance: f64,
) -> Pin<Box<dyn Future<Output = ()> + 'p>>
where
    W: Copy + std::fmt::Debug + Default,
    W: Add<Output = W> + AddAssign + Sub<Output = W> + Sum + PartialOrd,
    W: num::ToPrimitive,
{
    use tracing::Instrument as _;

    let fut = async move {
        if iter_count == 0 {
            // No need to split, all thoses items are in the same part.
            apply_part_id(items, iter_id);
            return;
        }
        if items.is_empty() {
            // No items left in this thread, remove it from global state.
            cancel_iterations(iter_ctxs, iter_count, iter_id).await;
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

fn rcb_thread<const D: usize, W>(
    iter_ctxs: &[IterationState<W>],
    items: &[Item<'_, D, W>],
    iter_count: usize,
    tolerance: f64,
) where
    W: Copy + std::fmt::Debug + Default,
    W: Add<Output = W> + AddAssign + Sub<Output = W> + Sum + PartialOrd,
    W: num::ToPrimitive,
{
    let copy_span = tracing::info_span!("copy items");
    let enter = copy_span.enter();

    let mut items = items.to_vec();

    mem::drop(enter);

    let task = rcb_iter(iter_ctxs, &mut items, iter_count, 0, 0, tolerance);
    futures_lite::future::block_on(task);
}

fn rcb<const D: usize, P, W>(
    partition: &mut [usize],
    points: P,
    weights: W,
    iter_count: usize,
    tolerance: f64,
) -> Result<(), Error>
where
    P: rayon::iter::IntoParallelIterator<Item = PointND<D>>,
    P::Iter: rayon::iter::IndexedParallelIterator,
    W: rayon::iter::IntoParallelIterator,
    W::Item: Copy + std::fmt::Debug + Default,
    W::Item: Add<Output = W::Item> + AddAssign + Sub<Output = W::Item> + Sum + PartialOrd,
    W::Item: num::ToPrimitive,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    let points = points.into_par_iter();
    let weights = weights.into_par_iter();

    if weights.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: weights.len(),
        });
    }
    if points.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: points.len(),
        });
    }
    if partition.is_empty() {
        // Would make the partition.min() at the end panic.
        return Ok(());
    }

    let init_span = tracing::info_span!("convert input and make initial data structures");
    let enter = init_span.enter();

    let mut items: Vec<_> = points
        .zip(weights)
        .zip(unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(&mut *partition) })
        .map(|((point, weight), part)| Item {
            point,
            weight,
            part,
        })
        .collect();
    let (items_per_thread, thread_count) =
        crate::work_share(items.len(), rayon::current_num_threads());
    let iteration_ctxs: Vec<_> = (0..usize::pow(2, iter_count as u32 + 1) - 1)
        .map(|_| IterationState::new(thread_count))
        .collect();

    mem::drop(enter);

    rayon::in_place_scope(|s| {
        for chunk in items.chunks_mut(items_per_thread) {
            let iteration_ctxs = &iteration_ctxs;
            s.spawn(move |_| rcb_thread(iteration_ctxs, chunk, iter_count, tolerance));
        }
    });

    let part_id_offset = *partition.par_iter().min().unwrap();
    partition
        .par_iter_mut()
        .for_each(|part_id| *part_id -= part_id_offset);

    Ok(())
}

fn simple_rcb_split<const D: usize, W>(
    items: &mut [Item<D, W>],
    coord: usize,
    tolerance: f64,
    mut min: f64,
    mut max: f64,
    sum: W,
) -> usize
where
    W: Copy + std::fmt::Debug + Default + Send + Sync,
    W: Add<Output = W> + AddAssign + Sub<Output = W> + Sum + PartialOrd,
    W: num::ToPrimitive,
{
    let mut prev_count_left = usize::MAX;
    loop {
        let split_target = (min + max) / 2.0;
        let (count_left, weight_left) = items
            .par_iter()
            .filter(|item| item.point[coord] < split_target)
            .fold(
                || (0, W::default()),
                |(count, weight), item| (count + 1, weight + item.weight),
            )
            .reduce(
                || (0, W::default()),
                |(count0, weight0), (count1, weight1)| (count0 + count1, weight0 + weight1),
            );

        let imbalance = {
            let ideal_weight_left = sum.to_f64().unwrap() / 2.0;
            let weight_left = weight_left.to_f64().unwrap();
            f64::abs((weight_left - ideal_weight_left) / ideal_weight_left)
        };
        if count_left == prev_count_left || imbalance < tolerance {
            return count_left;
        }
        prev_count_left = count_left;

        let weight_right = sum - weight_left;
        if weight_left < weight_right {
            min = split_target;
        } else {
            max = split_target;
        }
    }
}

fn simple_rcb_recurse<const D: usize, W>(
    items: &mut [Item<D, W>],
    iter_count: usize,
    iter_id: usize,
    coord: usize,
    tolerance: f64,
) where
    W: Copy + std::fmt::Debug + Default + Send + Sync,
    W: Add<Output = W> + AddAssign + Sub<Output = W> + Sum + PartialOrd,
    W: num::ToPrimitive,
{
    if items.is_empty() {
        // Would make min/max computation panic.
        return;
    }
    if iter_count == 0 {
        items
            .into_par_iter()
            .for_each(|item| item.part.store(iter_id, Ordering::Relaxed));
        return;
    }

    let sum: W = items.par_iter().map(|item| item.weight).sum();
    let (min, max) = items
        .par_iter()
        .fold(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(min, max), item| {
                (
                    f64::min(min, item.point[coord]),
                    f64::max(max, item.point[coord]),
                )
            },
        )
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(min0, max0), (min1, max1)| (f64::min(min0, min1), f64::max(max0, max1)),
        );

    let split_idx = simple_rcb_split(items, coord, tolerance, min, max, sum);
    let (left, right) = if split_idx == items.len() {
        items.split_at_mut(items.len())
    } else {
        let (left, _, _right_minus_one) = items
            .select_nth_unstable_by(split_idx, |item1, item2| {
                f64::partial_cmp(&item1.point[coord], &item2.point[coord]).unwrap()
            });
        let left_len = left.len();
        items.split_at_mut(left_len)
    };

    rayon::join(
        || {
            simple_rcb_recurse(
                left,
                iter_count - 1,
                2 * iter_id + 1,
                (coord + 1) % D,
                tolerance,
            )
        },
        || {
            simple_rcb_recurse(
                right,
                iter_count - 1,
                2 * iter_id + 2,
                (coord + 1) % D,
                tolerance,
            )
        },
    );
}

fn simple_rcb<const D: usize, P, W>(
    partition: &mut [usize],
    points: P,
    weights: W,
    iter_count: usize,
    tolerance: f64,
) -> Result<(), Error>
where
    P: rayon::iter::IntoParallelIterator<Item = PointND<D>>,
    P::Iter: rayon::iter::IndexedParallelIterator,
    W: rayon::iter::IntoParallelIterator,
    W::Item: Copy + std::fmt::Debug + Default + Send + Sync,
    W::Item: Add<Output = W::Item> + AddAssign + Sub<Output = W::Item> + Sum + PartialOrd,
    W::Item: num::ToPrimitive,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    let points = points.into_par_iter();
    let weights = weights.into_par_iter();

    if weights.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: weights.len(),
        });
    }
    if points.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: points.len(),
        });
    }
    if partition.is_empty() {
        // Would make the partition.min() at the end panic.
        return Ok(());
    }

    let mut items: Vec<_> = points
        .zip(weights)
        .zip(unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(&mut *partition) })
        .map(|((point, weight), part)| Item {
            point,
            weight,
            part,
        })
        .collect();

    simple_rcb_recurse(&mut items, iter_count, 0, 0, tolerance);

    let part_id_offset = *partition.par_iter().min().unwrap();
    partition
        .par_iter_mut()
        .for_each(|part_id| *part_id -= part_id_offset);

    Ok(())
}

/// # Recursive Coordinate Bisection algorithm
///
/// Partitions a mesh based on the nodes coordinates and coresponding weights.
///
/// This is the most simple and straightforward geometric algorithm. It operates
/// as follows for a N-dimensional set of points:
///
/// At each iteration, select a vector `n` of the canonical basis
/// `(e_0, ..., e_{n-1})`. Then, split the set of points with an hyperplane
/// orthogonal to `n`, such that the two parts of the splits are evenly
/// weighted. Finally, recurse by reapplying the algorithm to the two parts with
/// an other normal vector selection.
///
/// # Example
///
/// ```rust
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// let points = [
///     Point2D::new(1., 1.),
///     Point2D::new(-1., 1.),
///     Point2D::new(1., -1.),
///     Point2D::new(-1., -1.),
/// ];
/// let weights = [1; 4];
/// let mut partition = [0; 4];
///
/// // Generate a partition of 4 parts (2 splits).
/// coupe::Rcb { iter_count: 2, ..Default::default() }
///     .partition(&mut partition, (points, weights))
///     .unwrap();
///
/// // All points are in different parts.
/// for i in 0..4 {
///     for j in 0..4 {
///         if j == i {
///             continue
///         }
///         assert_ne!(partition[i], partition[j])
///     }
/// }
/// ```
///
/// # Reference
///
/// Berger, M. J. and Bokhari, S. H., 1987. A partitioning strategy for
/// nonuniform problems on multiprocessors. *IEEE Transactions on Computers*,
/// C-36(5):570–580. <doi:10.1109/TC.1987.1676942>.
#[derive(Default)]
pub struct Rcb {
    /// The number of iterations of the algorithm. This will yield a partition
    /// of at most `2^num_iter` parts.
    pub iter_count: usize,

    /// Tolerance on the normalized imbalance.
    pub tolerance: f64,

    /// Use a faster, experimental implementation of the algorithm.
    pub fast: bool,
}

impl<const D: usize, P, W> crate::Partition<(P, W)> for Rcb
where
    P: rayon::iter::IntoParallelIterator<Item = PointND<D>>,
    P::Iter: rayon::iter::IndexedParallelIterator,
    W: rayon::iter::IntoParallelIterator,
    W::Item: Copy + std::fmt::Debug + Default + Send + Sync,
    W::Item: Add<Output = W::Item> + AddAssign + Sub<Output = W::Item> + Sum + PartialOrd,
    W::Item: num::ToPrimitive,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (P, W),
    ) -> Result<Self::Metadata, Self::Error> {
        if part_ids.len() < 2 {
            // Would make Itertools::minmax().into_option() return None.
            return Ok(());
        }
        if self.fast {
            rcb(part_ids, points, weights, self.iter_count, self.tolerance)
        } else {
            simple_rcb(part_ids, points, weights, self.iter_count, self.tolerance)
        }
    }
}

// pub because it is also useful for multijagged and required for benchmarks
pub fn axis_sort<const D: usize>(
    points: &[PointND<D>],
    permutation: &mut [usize],
    current_coord: usize,
) {
    permutation.par_sort_unstable_by(|i1, i2| {
        if points[*i1][current_coord] < points[*i2][current_coord] {
            cmp::Ordering::Less
        } else {
            cmp::Ordering::Greater
        }
    })
}

fn rib<const D: usize, W>(
    partition: &mut [usize],
    points: &[PointND<D>],
    weights: W,
    n_iter: usize,
) -> Result<(), Error>
where
    Const<D>: DimSub<Const<1>>,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
    W: rayon::iter::IntoParallelIterator,
    W::Item: Copy + std::fmt::Debug + Default,
    W::Item: Add<Output = W::Item> + AddAssign + Sub<Output = W::Item> + Sum + PartialOrd,
    W::Item: num::ToPrimitive,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    let mbr = Mbr::from_points(points);
    let points = points.par_iter().map(|p| mbr.mbr_to_aabb(p));
    // When the rotation is done, we just apply RCB
    rcb(partition, points, weights, n_iter, 0.05)
}

/// # Recursive Inertial Bisection algorithm
///
/// Partitions a mesh based on the nodes coordinates and coresponding weights.
///
/// A variant of the [Recursive Coordinate Bisection algorithm][crate::Rcb]
/// where a basis change is performed beforehand so that the first coordinate of
/// the new basis is colinear to the inertia axis of the set of points. This has
/// the goal of producing better shaped partition than [RCB][crate::Rcb].
///
/// # Example
///
/// ```rust
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// // Here, the inertia axis is the y axis.
/// // We thus expect Rib to split horizontally first.
/// let points = [
///     Point2D::new(1., 10.),
///     Point2D::new(-1., 10.),
///     Point2D::new(1., -10.),
///     Point2D::new(-1., -10.),
/// ];
/// let weights = [1; 4];
/// let mut partition = [0; 4];
///
/// // Generate a partition of 2 parts (1 split).
/// coupe::Rib { iter_count: 1 }
///     .partition(&mut partition, (&points, weights))
///     .unwrap();
///
/// // The two points at the top are in the same part.
/// assert_eq!(partition[0], partition[1]);
///
/// // The two points at the bottom are in the same part.
/// assert_eq!(partition[2], partition[3]);
///
/// // There are two different parts.
/// assert_ne!(partition[1], partition[2]);
/// ```
///
/// # Reference
///
/// Williams, Roy D., 1991. Performance of dynamic load balancing algorithms for
/// unstructured mesh calculations. *Concurrency: Practice and Experience*,
/// 3(5):457–481. <doi:10.1002/cpe.4330030502>.
pub struct Rib {
    /// The number of iterations of the algorithm. This will yield a partition
    /// of at most `2^num_iter` parts.
    pub iter_count: usize,
}

impl<'a, const D: usize, W> crate::Partition<(&'a [PointND<D>], W)> for Rib
where
    Const<D>: DimSub<Const<1>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
    W: rayon::iter::IntoParallelIterator,
    W::Item: Copy + std::fmt::Debug + Default,
    W::Item: Add<Output = W::Item> + AddAssign + Sub<Output = W::Item> + Sum + PartialOrd,
    W::Item: num::ToPrimitive,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (&'a [PointND<D>], W),
    ) -> Result<Self::Metadata, Self::Error> {
        rib(part_ids, points, weights, self.iter_count)
    }
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
        let weights = [1.; 8];
        let points = [
            Point2D::from([-1.3, 6.]),
            Point2D::from([2., -4.]),
            Point2D::from([1., 1.]),
            Point2D::from([-3., -2.5]),
            Point2D::from([-1.3, -0.3]),
            Point2D::from([2., 1.]),
            Point2D::from([-3., 1.]),
            Point2D::from([1.3, -2.]),
        ];

        let mut partition = [0; 8];
        rayon::ThreadPoolBuilder::new()
            .num_threads(1) // make the test deterministic
            .build()
            .unwrap()
            .install(|| rcb(&mut partition, points, weights, 2, 0.05))
            .unwrap();

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

        let mut partition = vec![0; points.len()];
        rcb(&mut partition, points, weights.par_iter().cloned(), 3, 0.05).unwrap();

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
