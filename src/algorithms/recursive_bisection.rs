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
use std::collections::BTreeMap;
use std::mem;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::RwLock;
use std::sync::Weak;

// Needed for an exponentiation where the exponent is a usize and needs to be
// converted to a u32.
const _USIZE_LARGER_THAN_U32: &[()] = &[(); mem::size_of::<usize>() - mem::size_of::<u32>()];

#[derive(Debug)]
struct Item<'p, const D: usize> {
    point: PointND<D>,
    weight: f64,
    part: &'p mut ProcessUniqueId,
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

    /// InitData is everything that is computed beforehand by all threads.
    ///
    /// It contains the bouding box (min, max) and the sum of all weights.  The
    /// bounding box is updated at each iteration to compute the "split target".
    struct InitData {
        min: f64,
        max: f64,
        sum: f64,
    }

    /// LoopData is the result of each iteration.
    ///
    /// It is written by all threads, each threads decrements the TTL
    /// (time-to-live). The thread that obtains ttl==0 updates InitData's min
    /// and max, and decides whether to continue the iteration or not.
    struct LoopData {
        /// Weight on the left of the split target.  The weight on the right can
        /// be obtained through InitData's sum.
        weight_left: f64,

        /// The number of weights on the left of the split target.  The number
        /// on the right can be obtained through items's length.
        count_left: usize,

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
        init_data: RwLock<InitData>,
        loop_data: Mutex<LoopData>,
        barrier: Barrier,
        /// whether to stop the computation
        run: AtomicBool,
    }

    if num_iter == 0 {
        #[cfg(debug_assertions)]
        println!("num_iter=0,num_items={}", items.len());
        s.spawn(move |_| {
            let part = ProcessUniqueId::new();
            for item in items {
                *item.part = part;
            }
            q.lock().free_threads(1);
        });
        return 1;
    }

    let num_items = items.len();
    let chunk_size = usize::max((num_items + num_avail_threads - 1) / num_avail_threads, 64);
    let num_threads = (num_items + chunk_size - 1) / chunk_size;
    #[cfg(debug_assertions)]
    println!(
        "num_iter={},avail_threads={},num_threads={},num_items={},chunk_size={}",
        num_iter, num_avail_threads, num_threads, num_items, chunk_size,
    );

    let ctx = Arc::new(ThreadContext {
        init_data: RwLock::new(InitData {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
        }),
        loop_data: Mutex::new(LoopData {
            weight_left: 0.0,
            count_left: 0,
            count_left_max: items.len(),
            count_left_min: 0,
            ttl: num_threads,
        }),
        barrier: Barrier::new(num_threads),
        run: AtomicBool::new(true),
    });

    let items = Arc::new(items);
    // Used to wait for all threads to be shut down, before unwraping items.
    let (t_send, t_recv) = crossbeam_channel::bounded::<()>(0);

    #[cfg(debug_assertions)]
    let mut i = 0;

    for chunk_start in (0..num_items).step_by(chunk_size) {
        let q = Arc::clone(&q);
        let ctx = Arc::clone(&ctx);
        let items = Arc::clone(&items);
        let t_send = t_send.clone();
        let t_recv = t_recv.clone();
        s.spawn(move |_| {
            #[cfg(debug_assertions)]
            println!("compute {}: spawn", i);
            struct ThreadItem {
                point_coord: f64,
                weight: f64,
            }

            // Copy weights and coordinates into a hopefully nearer memory part.
            let chunk_end = usize::min(num_items, chunk_start + chunk_size);
            let thread_items: Vec<_> = items[chunk_start..chunk_end]
                .iter()
                .map(|item| ThreadItem {
                    point_coord: item.point[coord],
                    weight: item.weight,
                })
                .collect();

            // Compute the sum, the min and the max and merge them.
            let partial_sum: f64 = thread_items.iter().map(|item| item.weight).sum();

            use itertools::Itertools as _;
            let (partial_min, partial_max) = thread_items
                .iter()
                .map(|item| item.point_coord)
                .minmax()
                .into_option()
                .unwrap();

            {
                #[cfg(debug_assertions)]
                println!("compute {}: write sum/min/max", i);
                let mut d = ctx.init_data.write().unwrap();
                if partial_min < d.min {
                    d.min = partial_min;
                }
                if d.max < partial_max {
                    d.max = partial_max;
                }
                d.sum += partial_sum;
            }
            #[cfg(debug_assertions)]
            println!("compute {}: waiting for barrier", i);
            ctx.barrier.wait();
            #[cfg(debug_assertions)]
            println!("compute {}: woke up", i);

            let mut lead = false;
            let mut split_target = 0.0;
            while ctx.run.load(SeqCst) {
                let InitData { min, max, .. } = *ctx.init_data.try_read().unwrap();
                split_target = (min + max) / 2.0;
                let partial_weight_left: f64 = thread_items
                    .iter()
                    .filter(|item| item.point_coord < split_target)
                    .map(|item| item.weight)
                    .sum();
                let partial_left_count = thread_items
                    .iter()
                    .filter(|item| item.point_coord < split_target)
                    .count();

                {
                    #[cfg(debug_assertions)]
                    println!("compute {}: write weight_left/right", i);
                    let mut ld = ctx.loop_data.lock().unwrap();
                    ld.weight_left += partial_weight_left;
                    ld.count_left += partial_left_count;
                    let ttl = ld.ttl - 1;
                    if ttl == 0 {
                        #[cfg(debug_assertions)]
                        println!("compute {}: last write, computing next split_target", i);
                        let mut id = ctx.init_data.try_write().unwrap();
                        let weight_right = id.sum - ld.weight_left;
                        let max_imbalance = tolerance * id.sum;

                        if ld.weight_left < weight_right {
                            id.min = split_target;
                            ld.count_left_min = ld.count_left;
                        } else {
                            id.max = split_target;
                            ld.count_left_max = ld.count_left;
                        }

                        if f64::abs(ld.weight_left - weight_right) < max_imbalance
                            || ld.count_left_min == ld.count_left_max
                        {
                            #[cfg(debug_assertions)]
                            println!(
                                "compute {}: finish computation, split_target={}",
                                i, split_target,
                            );
                            ctx.run.store(false, SeqCst);
                            lead = true;
                        } else {
                            ld.weight_left = 0.0;
                            ld.count_left = 0;
                            ld.ttl = num_threads;
                        }
                    } else {
                        ld.ttl = ttl;
                    }
                }
                #[cfg(debug_assertions)]
                println!("compute {}: waiting for barrier", i);
                ctx.barrier.wait();
                #[cfg(debug_assertions)]
                println!("compute {}: woke up", i);
            }
            if lead {
                mem::drop(t_send);
                t_recv.recv().unwrap_err();
                let items = match Arc::try_unwrap(items) {
                    Ok(items) => items,
                    Err(_) => unreachable!(),
                };
                let (left, right): (Vec<_>, Vec<_>) = items
                    .into_iter()
                    .partition(|item| item.point[coord] < split_target);
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
                lock.free_threads(num_threads);
            }
        });
        #[cfg(debug_assertions)]
        {
            i += 1;
        }
    }
    #[cfg(debug_assertions)]
    assert_eq!(i, num_threads);

    num_threads
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
            let available_threads = self.available_threads.load(SeqCst);
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
                    self.available_threads.load(SeqCst),
                );
            }
            todo_list = self.cond.wait(todo_list).unwrap();
            self.todo_list = Some(todo_list);
        }
    }

    pub fn lock_threads(&self, used_threads: usize) {
        self.available_threads.fetch_sub(used_threads, SeqCst);
    }

    pub fn free_threads(&self, extra_threads: usize) {
        self.available_threads.fetch_add(extra_threads, SeqCst);
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

pub fn rcb<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    n_iter: usize,
) -> Vec<ProcessUniqueId> {
    let dummy_id = ProcessUniqueId::new();
    let mut partition = vec![dummy_id; points.len()];

    rayon::scope(|s| {
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
