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
use std::sync::RwLock;
use std::sync::Weak;

// Needed for an exponentiation where the exponent is a usize and needs to be
// converted to a u32.
const _USIZE_LARGER_THAN_U32: &[()] = &[(); mem::size_of::<usize>() - mem::size_of::<u32>()];

#[derive(Clone)]
struct Items<const D: usize> {
    points: Vec<PointND<D>>,
    weights: Vec<f64>,
}

struct Job<const D: usize> {
    items: Items<D>,
    coord: usize,
    num_iter: usize,
}

impl<const D: usize> std::fmt::Debug for Job<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Job {{ item_count: {}, coord: {}, num_iter: {} }}",
            self.items.points.len(),
            self.coord,
            self.num_iter,
        )
    }
}

use mpi::collective::Root as _;
use mpi::collective::SystemOperation;
use mpi::request::StaticScope;
use mpi::topology::Communicator as _;

fn rcb_root_iter<'p, const D: usize>(items: Items<D>, coord: usize) -> (Items<D>, Items<D>) {
    let world = mpi::topology::SystemCommunicator::world();
    let root = world.process_at_rank(0);

    println!("Ouais.");
    let points = items.points.as_slice();
    let points: &[f64] =
        unsafe { std::slice::from_raw_parts(points.as_ptr() as *const f64, points.len() * D) };

    let mut min = 0.0;
    root.reduce_into_root(points, &mut min, SystemOperation::min());
    let mut max = 0.0;
    root.reduce_into_root(points, &mut max, SystemOperation::max());
    let mut sum = 0.0;
    root.reduce_into_root(&items.weights, &mut sum, SystemOperation::sum());
    println!("Ouais!");

    let a = Items {
        points: Vec::new(),
        weights: Vec::new(),
    };
    (a.clone(), a)
}

fn rcb_nonroot_iter<const D: usize>() {
    let world = mpi::topology::SystemCommunicator::world();
    let root = world.process_at_rank(0);

    println!("Ah.");
    let lol: &mut [f64] = &mut [];
    root.reduce_into(lol, SystemOperation::min());
    root.reduce_into(lol, SystemOperation::max());
    root.reduce_into(lol, SystemOperation::sum());
    println!("Ah!");
}

fn rcb_root<'p, const D: usize>(items: Items<D>, num_iter: usize) {
    let world = mpi::topology::SystemCommunicator::world();
    let root = world.process_at_rank(0);

    let mut queue = Vec::new();
    queue.push(Job {
        items,
        coord: 0,
        num_iter,
    });

    while let Some(Job {
        items,
        coord,
        num_iter,
    }) = queue.pop()
    {
        println!("Iteration");
        root.broadcast_into(&mut true);
        let (left, right) = rcb_root_iter(items, coord);
        if num_iter == 0 {
            continue;
        }
        queue.push(Job {
            items: left,
            coord: (coord + 1) % D,
            num_iter: num_iter - 1,
        });
        queue.push(Job {
            items: right,
            coord: (coord + 1) % D,
            num_iter: num_iter - 1,
        });
    }

    root.broadcast_into(&mut false);
}

pub fn rcb<const D: usize>(points: &[PointND<D>], weights: &[f64], num_iter: usize) -> Vec<usize> {
    let _universe = mpi::initialize();
    let world = mpi::topology::SystemCommunicator::world();

    if world.rank() == 0 {
        let items = Items {
            points: points.to_vec(),
            weights: weights.to_vec(),
        };

        rcb_root(items, num_iter);

        Vec::new()
    } else {
        let root = world.process_at_rank(0);
        let mut run = true;
        loop {
            root.broadcast_into(&mut run);
            if !run {
                break;
            }
            rcb_nonroot_iter::<D>();
        }
        Vec::new()
    }
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
