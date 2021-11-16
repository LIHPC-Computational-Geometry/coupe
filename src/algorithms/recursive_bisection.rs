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
use std::collections::BTreeMap;
use std::mem;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::Weak;

// Needed for an exponentiation where the exponent is a usize and needs to be
// converted to a u32.
const _USIZE_LARGER_THAN_U32: &[()] = &[(); mem::size_of::<usize>() - mem::size_of::<u32>()];

fn parallel_chunks<'a, F, T>(
    s: &rayon::Scope<'a>,
    items: Vec<T>,
    thread_count: usize,
    f: F,
) -> usize
where
    T: 'a + Clone + Send + Sync,
    F: 'a + Fn(Vec<T>) + Send + Sync,
{
    let items_len = items.len();
    let chunk_size = usize::max((items_len + thread_count - 1) / thread_count, 64);
    let used_thread_count = (items_len + chunk_size - 1) / chunk_size;

    debug_assert_eq!(
        used_thread_count,
        (0..items_len).step_by(chunk_size).count(),
    );

    let items = Arc::new(items);
    let f = Arc::new(f);

    for chunk_start in (0..items_len).step_by(chunk_size) {
        let items = Arc::clone(&items);
        let f = Arc::clone(&f);
        s.spawn(move |_| {
            // Copy weights and coordinates into a hopefully nearer memory part.
            let chunk_end = usize::min(items_len, chunk_start + chunk_size);
            let thread_items: Vec<T> = items[chunk_start..chunk_end].iter().cloned().collect();

            f(thread_items)
        });
    }

    used_thread_count
}

#[derive(Clone, Debug)]
struct Item<'p, const D: usize> {
    point: PointND<D>,
    weight: f64,
    part: &'p AtomicUsize,
}

fn weighted_median<'p, const D: usize>(
    s: &rayon::Scope<'p>,
    q: Arc<JobQueue<'p, D>>,
    items: Vec<Item<'p, D>>,
    coord: usize,
    num_iter: usize,
    tolerance: f64,
    num_avail_threads: usize,
) -> usize {
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

        num_threads: usize,
    }

    /// Data shared by all threads
    struct ThreadContext {
        c: Condvar,
        data: Mutex<Locked>,

        barrier: Barrier,
        /// whether to stop the computation
        run: AtomicBool,
    }

    if num_iter == 0 {
        #[cfg(debug_assertions)]
        println!("num_iter=0,num_items={}", items.len());
        s.spawn(move |_| {
            let part = crate::uid();
            for item in items {
                item.part.store(part, Ordering::Relaxed);
            }
            q.lock().free_threads(1);
        });
        return 1;
    }

    let num_items = items.len();
    let chunk_size = usize::max((num_items + num_avail_threads - 1) / num_avail_threads, 64);
    let num_threads = (num_items + chunk_size - 1) / chunk_size;

    let ctx = ThreadContext {
        c: Condvar::new(),
        data: Mutex::new(Locked {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            left_part_weight: 0.0,
            additional_left_weight: 0.0,
            left_part_count: 0,
            count_left_max: items.len(),
            count_left_min: 0,
            ttl: num_threads,
            num_threads,
        }),
        barrier: Barrier::new(num_threads),
        run: AtomicBool::new(true),
    };

    let (t_send, t_recv) = crossbeam_channel::bounded::<(Vec<Item<'p, D>>, usize)>(num_threads - 1);

    parallel_chunks(s, items, num_avail_threads, move |mut thread_items| {
        // Compute the sum, the min and the max and merge them.
        let partial_sum: f64 = thread_items.iter().map(|item| item.weight).sum();

        let (partial_min, partial_max) = thread_items
            .iter()
            .map(|item| item.point[coord])
            .minmax()
            .into_option()
            .unwrap();

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
            min = d.min;
            max = d.max;
        }
        if !ctx.barrier.wait().is_leader() {
            let d = ctx.data.lock().unwrap();
            min = d.min;
            max = d.max;
        }

        let mut lead = false;
        let mut item_view = &mut *thread_items;
        let mut additional_partial_left_count = 0;
        let mut median_idx = 0;
        while ctx.run.load(Ordering::SeqCst) {
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

            let mut data = ctx.data.lock().unwrap();

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
                    data.num_threads -= 1;
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
            let max_imbalance = tolerance * data.sum;

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
                data = ctx
                    .c
                    .wait_while(data, |data| data.max == max && data.min == min)
                    .unwrap();
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
                    data.num_threads -= 1;
                    data.ttl = data.ttl.checked_sub(1).unwrap();
                    break;
                }
                continue;
            }

            if f64::abs(weight_left - remaining_weight) < max_imbalance
                || data.count_left_min == data.count_left_max
            {
                lead = ctx
                    .run
                    .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok(); // is always ok for now, since we're inside a lock
            }

            if item_view.is_empty() {
                data.num_threads -= 1;
            }

            data.left_part_count = 0;
            data.left_part_weight = 0.0;
            data.ttl = data.num_threads;

            ctx.c.notify_all();

            if item_view.is_empty() {
                break;
            }
        }
        let median_idx = additional_partial_left_count + median_idx;
        if !lead {
            // run can still be true if all threads exited early
            lead = ctx
                .run
                .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok();
        }
        if lead {
            let mut thread_results = Vec::with_capacity(num_threads);
            thread_results.push((thread_items, median_idx));
            for _ in 0..num_threads - 1 {
                thread_results.push(t_recv.recv().unwrap());
            }
            let data = ctx.data.try_lock().unwrap();
            let mut left = Vec::with_capacity(data.left_part_count);
            let mut right = Vec::with_capacity(num_items - data.left_part_count);
            for (mut thread_items, median_idx) in thread_results {
                right.extend(thread_items.drain(median_idx..));
                left.extend(thread_items);
            }
            let next_coord = (coord + 1) % D;
            let (threads_left, threads_right) = {
                let now = num_avail_threads as f64;
                let left = left.len() as f64;
                let right = right.len() as f64;
                let threads_left = left / (left + right) * now;
                let threads_right = right / (left + right) * now;
                (threads_left as usize, threads_right as usize)
            };
            let mut lock = q.lock();
            lock.push(
                usize::max(threads_left, 1),
                Job {
                    items: left,
                    num_iter: num_iter - 1,
                    coord: next_coord,
                },
            );
            lock.push(
                usize::max(threads_right, 1),
                Job {
                    items: right,
                    num_iter: num_iter - 1,
                    coord: next_coord,
                },
            );
            lock.free_threads(1);
        } else {
            t_send.send((thread_items, median_idx)).unwrap();
            q.lock().free_threads(1);
        }
    })
}

struct Job<'p, const D: usize> {
    items: Vec<Item<'p, D>>,
    coord: usize,
    num_iter: usize,
}

impl<'p, const D: usize> std::fmt::Debug for Job<'p, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Job {{ item_count: {}, coord: {}, num_iter: {} }}",
            self.items.len(),
            self.coord,
            self.num_iter,
        )
    }
}

type TodoList<'p, const D: usize> = BTreeMap<usize, Vec<Job<'p, D>>>;

struct JobQueue<'p, const D: usize> {
    available_threads: AtomicUsize,
    cond: Condvar,
    todo_list: Mutex<TodoList<'p, D>>,
}

impl<'p, const D: usize> JobQueue<'p, D> {
    pub fn new(available_threads: usize) -> Arc<JobQueue<'p, D>> {
        Arc::new(JobQueue {
            available_threads: AtomicUsize::new(available_threads),
            cond: Condvar::new(),
            todo_list: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn lock<'q>(self: &'q Arc<JobQueue<'p, D>>) -> JobQueueLock<'p, 'q, D> {
        JobQueueLock {
            queue_ref: Arc::downgrade(self),
            available_threads: &self.available_threads,
            cond: &self.cond,
            todo_list: Some(self.todo_list.lock().unwrap()),
        }
    }
}

struct JobQueueLock<'p, 'q, const D: usize> {
    queue_ref: Weak<JobQueue<'p, D>>,
    available_threads: &'q AtomicUsize,
    cond: &'q Condvar,
    todo_list: Option<MutexGuard<'q, TodoList<'p, D>>>,
}

impl<'p, 'q, const D: usize> JobQueueLock<'p, 'q, D> {
    pub fn push(&mut self, job_threads: usize, job: Job<'p, D>) {
        self.todo_list
            .as_mut()
            .unwrap()
            .entry(job_threads)
            .or_default()
            .push(job);
    }

    pub fn pop(&mut self) -> Option<(usize, Job<'p, D>)> {
        loop {
            let available_threads = self.available_threads.load(Ordering::SeqCst);
            for (num_threads, bucket) in self
                .todo_list
                .as_mut()
                .unwrap()
                .range_mut(..=available_threads)
            {
                if let Some(job) = bucket.pop() {
                    return Some((*num_threads, job));
                }
            }
            if self.queue_ref.strong_count() <= 1 {
                return None;
            }
            let mut todo_list = mem::replace(&mut self.todo_list, None).unwrap();
            #[cfg(debug_assertions)]
            if !todo_list.values().all(Vec::is_empty) {
                println!(
                    "Queue::pop: self.todo_list={:?}, available_threads={}",
                    todo_list,
                    self.available_threads.load(Ordering::SeqCst),
                );
            }
            todo_list = self.cond.wait(todo_list).unwrap();
            self.todo_list = Some(todo_list);
        }
    }

    pub fn lock_threads(&self, used_threads: usize) {
        self.available_threads
            .fetch_sub(used_threads, Ordering::SeqCst);
    }

    pub fn free_threads(&self, extra_threads: usize) {
        self.available_threads
            .fetch_add(extra_threads, Ordering::SeqCst);
    }
}

impl<'p, 'q, const D: usize> Drop for JobQueueLock<'p, 'q, D> {
    fn drop(&mut self) {
        self.cond.notify_all();
    }
}

fn rcb_recurse<'p, const D: usize>(
    s: &rayon::Scope<'p>,
    items: Vec<Item<'p, D>>,
    num_iter: usize,
    coord: usize,
    tolerance: f64,
    available_threads: usize,
) {
    let q = JobQueue::new(available_threads);
    q.lock().push(
        available_threads,
        Job {
            items,
            coord,
            num_iter,
        },
    );

    let total_jobs = usize::pow(2, num_iter as u32 + 1) - 1;
    for job_num in 0..total_jobs {
        #[cfg(debug_assertions)]
        println!("Waiting for job #{} on {}", job_num, total_jobs);
        let mut lock = q.lock();
        let (num_threads, job) = match lock.pop() {
            Some(val) => val,
            None => break,
        };
        let Job {
            items,
            coord,
            num_iter,
        } = job;
        let used_threads = weighted_median(
            s,
            Arc::clone(&q),
            items,
            coord,
            num_iter,
            tolerance,
            num_threads,
        );
        lock.lock_threads(used_threads);
    }
}

pub fn rcb<const D: usize>(points: &[PointND<D>], weights: &[f64], n_iter: usize) -> Vec<usize> {
    let mut partition = vec![0; points.len()];

    rayon::in_place_scope(|s| {
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
        rcb_recurse(s, items, n_iter, 0, 0.05, rayon::current_num_threads() - 1);
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
