//! An implementation of the Hilbert space filling curve.
//!
//! With this technique, a set of 2D points (p0, ..., pn) is mapped to a set of numbers (i1, ..., in)
//! used to reorder the set of points. How the mapping is defined follows how encoding the Hilbert curve is
//! described in "Encoding and Decoding the Hilbert Order" by XIAN LIU and GÜNTHER SCHRACK
//!
//! The hilbert curve depends on a grid resolution called `order`. Basically,
//! the minimal bounding rectangle of the set of points is split in 2^(2*order) cells.
//! All the points in a given cell will have the same encoding.
//!
//! The complexity of encoding a point is O(order)

use crate::geometry::{Mbr, Point2D};
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

pub fn hilbert_curve_partition(
    points: &[Point2D],
    weights: &[f64],
    num_partitions: usize,
    order: usize,
) -> Vec<ProcessUniqueId> {
    assert!(
        order < 32,
        "Cannot construct a Hilbert curve of order >= 32 because 2^32 would overflow u32 capacity."
    );

    let compute_hilbert_index = hilbert_index_computer(points, order);
    let mut permutation = (0..points.len()).into_par_iter().collect::<Vec<_>>();
    let hilbert_indices = points
        .par_iter()
        .map(|p| compute_hilbert_index((p.x, p.y)))
        .collect::<Vec<_>>();

    permutation
        .as_mut_slice()
        .par_sort_by_key(|idx| hilbert_indices[*idx]);

    // dummy modifiers to use directly the routine from multi_jagged
    let modifiers = vec![1. / num_partitions as f64; num_partitions];

    let initial_id = ProcessUniqueId::new();
    let mut partition = vec![initial_id; points.len()];

    let split_positions = super::multi_jagged::compute_split_positions(
        weights,
        &permutation,
        num_partitions - 1,
        &modifiers,
    );

    let mut sub_permutation =
        super::multi_jagged::split_at_mut_many(&mut permutation, &split_positions);
    let atomic_partition_handle = std::sync::atomic::AtomicPtr::new(partition.as_mut_ptr());

    sub_permutation.par_iter_mut().for_each(|slice| {
        let part_id = ProcessUniqueId::new();
        let ptr = atomic_partition_handle.load(std::sync::atomic::Ordering::Relaxed);
        for i in slice.iter_mut() {
            unsafe { std::ptr::write(ptr.add(*i), part_id) }
        }
    });

    partition
}

#[allow(unused)]
pub(crate) fn hilbert_curve_reorder(points: &[Point2D], order: usize) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..points.len()).into_par_iter().collect();
    hilbert_curve_reorder_permu(points, &mut permutation, order);
    permutation
}

/// Reorder a set of points and weights following the hilbert curve technique.
/// First, the minimal bounding rectangle of the set of points is computed and local
/// coordinated are defined on it: the mbr is seen as [0; 2^order - 1]^2.
/// Then the hilbert curve is computed from those local coordinates.
pub(crate) fn hilbert_curve_reorder_permu(
    points: &[Point2D],
    permutation: &mut [usize],
    order: usize,
) {
    assert!(
        order < 32,
        "Cannot construct a Hilbert curve of order >= 32 because 2^32 would overflow u32 capacity."
    );

    let compute_hilbert_index = hilbert_index_computer(points, order);

    permutation.par_sort_by_key(|idx| {
        let p = points[*idx];
        compute_hilbert_index((p.x, p.y))
    });
}

fn hilbert_index_computer(points: &[Point2D], order: usize) -> impl Fn((f64, f64)) -> u32 {
    let mbr = Mbr::from_points(&points);

    let (ax, ay) = {
        let aabb = mbr.aabb();
        (
            (aabb.p_min().x, aabb.p_max().x),
            (aabb.p_min().y, aabb.p_max().y),
        )
    };

    let rotate = move |p: &Point2D| mbr.mbr_to_aabb(p);

    let x_mapping = segment_to_segment(ax.0, ax.1, 0., order as f64);
    let y_mapping = segment_to_segment(ay.0, ay.1, 0., order as f64);

    move |p| {
        let p = rotate(&Point2D::new(p.0, p.1));
        encode(x_mapping(p.x) as u32, y_mapping(p.y) as u32, order)
    }
}

fn encode(x: u32, y: u32, order: usize) -> u32 {
    assert!(
        order < 32,
        "Cannot construct a Hilbert curve of order >= 32 because 2^32 would overflow u32 capacity."
    );
    assert!(
        x < 2u32.pow(order as u32),
        "Cannot encode the point {:?} on an hilbert curve of order {} because x >= 2^order.",
        (x, y),
        order,
    );
    assert!(
        y < 2u32.pow(order as u32),
        "Cannot encode the point {:?} on an hilbert curve of order {} because y >= 2^order.",
        (x, y),
        order,
    );

    let mask = (1 << order) - 1;
    let h_even = x ^ y;
    let not_x = !x & mask;
    let not_y = !y & mask;
    let temp = not_x ^ y;

    let mut v0 = 0;
    let mut v1 = 0;

    for _ in 1..order {
        v1 = ((v1 & h_even) | ((v0 ^ not_y) & temp)) >> 1;
        v0 = ((v0 & (v1 ^ not_x)) | (!v0 & (v1 ^ not_y))) >> 1;
    }

    let h_odd = (!v0 & (v1 ^ x)) | (v0 & (v1 ^ not_y));

    interleave_bits(h_odd, h_even)
}

fn interleave_bits(odd: u32, even: u32) -> u32 {
    let mut val = 0;
    let mut max = odd.max(even);
    let mut n = 0;
    while max > 0 {
        n += 1;
        max >>= 1;
    }

    for i in 0..n {
        let mask = 1 << i;
        let a = if (even & mask) > 0 { 1 << (2 * i) } else { 0 };
        let b = if (odd & mask) > 0 {
            1 << (2 * i + 1)
        } else {
            0
        };
        val += a + b;
    }

    val
}

// Compute a mapping from [a_min; a_max] to [b_min; b_max]
fn segment_to_segment(a_min: f64, a_max: f64, b_min: f64, b_max: f64) -> impl Fn(f64) -> f64 {
    assert!(
        a_min <= a_max,
        "Cannot construct a segment to segment mapping because a_max < a_min. a_min = {}, a_max = {}.",
        a_min,
        a_max,
    );
    assert!(
        b_min <= b_max,
        "Cannot construct a segment to segment mapping because b_max < b_min. b_min = {}, b_max = {}.",
        b_min,
        b_max,
    );

    let da = a_min - a_max;
    let db = b_min - b_max;
    let alpha = db / da;
    let beta = b_min - a_min * alpha;

    move |x| {
        assert!(
            a_min <= x && x <= a_max,
            "Called a mapping from [{}, {}] to [{}, {}] with the invalid value {}.",
            a_min,
            a_max,
            b_min,
            b_max,
            x,
        );
        alpha * x + beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_segment_to_segment() {
        let a_min = -1.;
        let a_max = 1.;
        let b_min = 0.;
        let b_max = 10.;

        let mapping = segment_to_segment(a_min, a_max, b_min, b_max);
        let inverse = segment_to_segment(b_min, b_max, a_min, a_max);

        assert_ulps_eq!(mapping(a_min), b_min);
        assert_ulps_eq!(mapping(a_max), b_max);
        assert_ulps_eq!(mapping(0.), 5.);
        assert_ulps_eq!(mapping(-0.5), 2.5);

        assert_ulps_eq!(inverse(b_min), a_min);
        assert_ulps_eq!(inverse(b_max), a_max);
        assert_ulps_eq!(inverse(5.), 0.);
        assert_ulps_eq!(inverse(2.5), -0.5);
    }

    #[test]
    #[should_panic]
    fn test_segment_to_segment_wrong_input() {
        let a_min = -1.;
        let a_max = 1.;
        let b_min = 0.;
        let b_max = 10.;

        let _mapping = segment_to_segment(a_max, a_min, b_min, b_max);
    }

    #[test]
    fn test_hilbert_curve_1() {
        let points = vec![(0, 0), (1, 1), (1, 0), (0, 1)];
        let indices = points
            .into_iter()
            .map(|(x, y)| encode(x, y, 1))
            .collect::<Vec<_>>();

        assert_eq!(indices, vec![0, 2, 3, 1]);
    }

    #[test]
    fn test_hilbert_curve_2() {
        let points = vec![
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 2),
            (3, 1),
            (2, 1),
            (2, 0),
            (3, 0),
        ];

        let expected: Vec<_> = (0..16).collect();
        let indices = points
            .into_iter()
            .map(|(x, y)| encode(x, y, 2))
            .collect::<Vec<_>>();

        assert_eq!(indices, expected);
    }
}
