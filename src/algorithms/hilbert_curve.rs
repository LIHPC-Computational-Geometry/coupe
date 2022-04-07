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
use std::fmt;

fn hilbert_curve_partition(
    partition: &mut [usize],
    points: &[Point2D],
    weights: &[f64],
    part_count: usize,
    order: usize,
) {
    debug_assert!(order < 64);

    let compute_hilbert_index = hilbert_index_computer(points, order);
    let mut permutation = (0..points.len()).into_par_iter().collect::<Vec<_>>();
    let hilbert_indices = points
        .par_iter()
        .map(compute_hilbert_index)
        .collect::<Vec<_>>();

    permutation
        .as_mut_slice()
        .par_sort_by_key(|idx| hilbert_indices[*idx]);

    // dummy modifiers to use directly the routine from multi_jagged
    let modifiers = vec![1. / part_count as f64; part_count];

    let split_positions =
        super::multi_jagged::compute_split_positions(weights, &permutation, &modifiers);

    let sub_permutation =
        super::multi_jagged::split_at_mut_many(&mut permutation, &split_positions);
    let atomic_partition_handle = std::sync::atomic::AtomicPtr::new(partition.as_mut_ptr());

    sub_permutation
        .par_iter()
        .enumerate()
        .for_each(|(part_id, slice)| {
            let ptr = atomic_partition_handle.load(std::sync::atomic::Ordering::Relaxed);
            for i in &**slice {
                unsafe { std::ptr::write(ptr.add(*i), part_id) }
            }
        });
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
fn hilbert_curve_reorder_permu(points: &[Point2D], permutation: &mut [usize], order: usize) {
    debug_assert!(order < 32);

    let compute_hilbert_index = hilbert_index_computer(points, order);

    permutation.par_sort_by_key(|idx| {
        let p = &points[*idx];
        compute_hilbert_index(p)
    });
}

/// Compute a mapping from [min; max] to [0; 2**order-1]
fn segment_to_segment(min: f64, max: f64, order: usize) -> impl Fn(f64) -> u64 {
    debug_assert!(min <= max);

    let width = max - min;
    let n = (1_u64 << order) as f64;
    let mut f = n / width;

    // Map max to (2**order-1).
    while n <= width * f {
        f = crate::nextafter(f, 0.0);
    }

    move |v| {
        debug_assert!(min <= v && v <= max, "{v} not in [{min};{max}]");
        (f * (v - min)) as u64
    }
}

fn hilbert_index_computer(points: &[Point2D], order: usize) -> impl Fn(&Point2D) -> u64 {
    let mbr = Mbr::from_points(points);
    let aabb = mbr.aabb();
    let p_min = aabb.p_min();
    let p_max = aabb.p_max();
    let x_mapping = segment_to_segment(p_min.x, p_max.x, order);
    let y_mapping = segment_to_segment(p_min.y, p_max.y, order);
    move |p| {
        let p = mbr.mbr_to_aabb(p);
        encode(x_mapping(p.x), y_mapping(p.y), order)
    }
}

/// Slower version of [encode], used to build the lookup table used by [encode].
///
/// This version takes the initial configuration as argument and also returns
/// the final configuration.
///
/// Taken from Marot, Célestin. "Parallel tetrahedral mesh generation." Prom:
/// Remacle, Jean-François <http://hdl.handle.net/2078.1/240626>.
///
/// TODO: once const-fn are more mature, take the "order" argument into account,
/// though it is only set to 6 for the purpose of building the lookup table.
const fn encode_slow(zorder: u64, _order: usize, mut config: usize) -> (u64, usize) {
    // BASE_PATTERN[i][j] is the hilbert index given:
    // - i: the current configuration,
    // - j: the quadrant in row-major order.
    const BASE_PATTERN: [[u64; 4]; 4] = [
        [0, 1, 3, 2], // config 0 = [1,2]   )
        [0, 3, 1, 2], // config 1 = [2,1]   n
        [2, 3, 1, 0], // config 2 = [-1,-2] (
        [2, 1, 3, 0], // config 3 = [-2,-1] U
    ];
    // CONFIGURATION[i][j] is the next configuration given:
    // - i: the current configuration,
    // - j: the quadrant in row-major order.
    const CONFIGURATION: [[usize; 4]; 4] = [
        [1, 0, 3, 0], // ) => U)
        //                    n)
        [0, 2, 1, 1], // n => nn
        //                    )(
        [2, 1, 2, 3], // ( => (U
        //                    (n
        [3, 3, 0, 2], // U => )(
                      //      UU
    ];

    let mut hilbert = 0;

    // TODO replace unrolled loop by "for i in (0..order).rev()" once for loops
    // are allowed in const fns.
    let mut i = 5;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];

    (hilbert, config)
}

const HILBERT_LUT: [u16; 16_384] = {
    let mut lut = [0; 16_384];
    let mut i: usize = 0;
    while i < 16_384 {
        let zorder = (i & 0xfff) as u64;
        let config = i >> 12;
        let (hilbert_order, config) = encode_slow(zorder, 6, config);
        lut[i] = (config << 12) as u16 | hilbert_order as u16;
        i += 1;
    }
    lut
};

fn encode(x: u64, y: u64, order: usize) -> u64 {
    debug_assert!(order < 64);
    debug_assert!(
        x < (1 << order),
        "Cannot encode the point {:?} on an hilbert curve of order {} because x >= 2^order.",
        (x, y),
        order,
    );
    debug_assert!(
        y < (1 << order),
        "Cannot encode the point {:?} on an hilbert curve of order {} because y >= 2^order.",
        (x, y),
        order,
    );

    let zorder = unsafe {
        std::arch::x86_64::_pdep_u64(x, 0x5555_5555_5555_5555 << 1)
            | std::arch::x86_64::_pdep_u64(y, 0x5555_5555_5555_5555)
    };

    let mut config: u16 = 0;
    let mut hilbert: u64 = 0;
    let mut shift: i64 = 2 * order as i64 - 12;
    while shift > 0 {
        config = HILBERT_LUT[((config & !0xfff) | ((zorder >> shift) & 0xfff) as u16) as usize];
        hilbert = (hilbert << 12) | (config & 0xfff) as u64;
        shift -= 12;
    }

    config =
        HILBERT_LUT[((config & !0xfff) | ((zorder << (-shift) as u64) & 0xfff) as u16) as usize];
    hilbert = (hilbert << 12) | (config & 0xfff) as u64;

    hilbert >> -shift
}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Invalid space filling curve order.
    InvalidOrder { max: u32, actual: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidOrder { max, actual } => {
                write!(
                    f,
                    "given hilbert curve order too high. Got {}, max={}.",
                    actual, max
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// # Hilbert space-filling curve algorithm
///
/// Projects points on the hilbert curve and splits this curve into a given
/// amount of parts.
///
/// The hilbert curve depends on a grid resolution called `order`. Basically,
/// the minimal bounding rectangle of the set of points is split into
/// `2^order * 2^order` cells.  All the points in a given cell will have the
/// same encoding.
///
/// The complexity of encoding a point is `O(order)`.
///
/// # Example
///
/// ```rust
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// let points = [
///     Point2D::new(0., 0.),
///     Point2D::new(1., 1.),
///     Point2D::new(0., 10.),
///     Point2D::new(1., 9.),
///     Point2D::new(9., 1.),
///     Point2D::new(10., 0.),
///     Point2D::new(10., 10.),
///     Point2D::new(9., 9.),
/// ];
/// let weights = [1.0; 8];
/// let mut partition = [0; 8];
///
/// // generate a partition of 4 parts
/// coupe::HilbertCurve { part_count: 4, order: 5 }
///     .partition(&mut partition, (points, weights))
///     .unwrap();
///
/// assert_eq!(partition[0], partition[1]);
/// assert_eq!(partition[2], partition[3]);
/// assert_eq!(partition[4], partition[5]);
/// assert_eq!(partition[6], partition[7]);
/// ```
///
/// # Reference
///
/// Marot, Célestin. *Parallel tetrahedral mesh generation*. Prom.: Remacle,
/// Jean-François <http://hdl.handle.net/2078.1/240626>.
pub struct HilbertCurve {
    pub part_count: usize,
    pub order: u32,
}

// hilbert curve is only implemented in 2d for now
impl<P, W> crate::Partition<(P, W)> for HilbertCurve
where
    P: AsRef<[Point2D]>,
    W: AsRef<[f64]>,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (P, W),
    ) -> Result<Self::Metadata, Self::Error> {
        if self.order >= 64 {
            return Err(Error::InvalidOrder {
                max: 63,
                actual: self.order,
            });
        }
        hilbert_curve_partition(
            part_ids,
            points.as_ref(),
            weights.as_ref(),
            self.part_count,
            self.order as usize,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_to_segment() {
        let mapping = segment_to_segment(0.0, 8.0, 3);

        assert_eq!(mapping(0.0), 0);
        assert_eq!(mapping(1.0), 0);
        assert_eq!(mapping(2.0), 1);
        assert_eq!(mapping(3.0), 2);
        assert_eq!(mapping(4.0), 3);
        assert_eq!(mapping(5.0), 4);
        assert_eq!(mapping(6.0), 5);
        assert_eq!(mapping(7.0), 6);
        assert_eq!(mapping(8.0), 7);

        assert_eq!(mapping(crate::nextafter(1.0, f64::INFINITY)), 1);
        assert_eq!(mapping(crate::nextafter(2.0, f64::INFINITY)), 2);
        assert_eq!(mapping(crate::nextafter(3.0, f64::INFINITY)), 3);
        assert_eq!(mapping(crate::nextafter(4.0, f64::INFINITY)), 4);
        assert_eq!(mapping(crate::nextafter(5.0, f64::INFINITY)), 5);
        assert_eq!(mapping(crate::nextafter(6.0, f64::INFINITY)), 6);
        assert_eq!(mapping(crate::nextafter(7.0, f64::INFINITY)), 7);
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
