use super::Error;
use crate::geometry::OrientedBoundingBox;
use crate::geometry::PointND;
use crate::BoundingBox;
use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use nalgebra::ToTypenum;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::cmp;
use std::iter::Sum;
use std::mem::MaybeUninit;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

/// Taken from the stdlib
///
/// TODO remove this once stabilized.
/// Tracking issue: <https://github.com/rust-lang/rust/issues/96097>
///
/// # Safety
///
/// Do not call with unhabited types.
unsafe fn array_assume_init<T, const N: usize>(array: [MaybeUninit<T>; N]) -> [T; N] {
    // SAFETY:
    // * The caller guarantees that all elements of the array are initialized
    // * `MaybeUninit<T>` and T are guaranteed to have the same layout
    // * `MaybeUninit` does not drop, so there are no double-frees
    // And thus the conversion is safe
    //std::intrinsics::assert_inhabited::<[T; N]>();
    (&array as *const _ as *const [T; N]).read()
}

fn array_init<F, T, const N: usize>(mut f: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    unsafe {
        let mut output: [MaybeUninit<T>; N] = MaybeUninit::uninit().assume_init();
        for (i, u) in output.iter_mut().enumerate() {
            u.write(f(i));
        }
        array_assume_init(output)
    }
}

fn array_map_mut<'a, F, T, U, const N: usize>(input: &'a mut [T; N], mut map: F) -> [U; N]
where
    F: FnMut(&'a mut T) -> U,
    U: 'a,
{
    unsafe {
        let mut output: [MaybeUninit<U>; N] = MaybeUninit::uninit().assume_init();
        for (u, t) in output.iter_mut().zip(input) {
            u.write(map(t));
        }
        array_assume_init(output)
    }
}

fn array_unzip<F, T, U, V, const N: usize>(input: [T; N], mut map: F) -> ([U; N], [V; N])
where
    F: FnMut(T) -> (U, V),
{
    unsafe {
        let mut output1: [MaybeUninit<U>; N] = MaybeUninit::uninit().assume_init();
        let mut output2: [MaybeUninit<V>; N] = MaybeUninit::uninit().assume_init();
        for ((u, v), t) in output1.iter_mut().zip(&mut output2).zip(input) {
            let (new_u, new_v) = map(t);
            u.write(new_u);
            v.write(new_v);
        }
        (array_assume_init(output1), array_assume_init(output2))
    }
}

fn array_unzip_mut<'a, F, T, U, V, const N: usize>(
    input: &'a mut [T; N],
    mut map: F,
) -> ([U; N], [V; N])
where
    F: FnMut(&'a mut T) -> (U, V),
    U: 'a,
    V: 'a,
{
    unsafe {
        let mut output1: [MaybeUninit<U>; N] = MaybeUninit::uninit().assume_init();
        let mut output2: [MaybeUninit<V>; N] = MaybeUninit::uninit().assume_init();
        for ((u, v), t) in output1.iter_mut().zip(&mut output2).zip(input) {
            let (new_u, new_v) = map(t);
            u.write(new_u);
            v.write(new_v);
        }
        (array_assume_init(output1), array_assume_init(output2))
    }
}

struct Items<'a, const D: usize, W> {
    points: [&'a mut [f32]; D],
    weights: &'a mut [W],
    parts: &'a mut [&'a AtomicUsize],
}

/// Return value of [rcb_split] and [par_rcb_split].
struct SplitResult<'a, const D: usize, W> {
    left: Items<'a, D, W>,
    right: Items<'a, D, W>,
    /// Weight of the left part, used to compute the sum for the next iteration.
    weight_left: W,
    /// Coordinate value of the split, used to compute the [BoundingBox]es.
    split_pos: f32,
}

/// Scalar version of [reorder_split].
fn reorder_split_scalar<const D: usize, W>(
    mut items: Items<'_, D, W>,
    pivot: usize,
    coord: usize,
) -> (Items<'_, D, W>, Items<'_, D, W>) {
    for ps in &mut items.points {
        ps.swap(0, pivot);
    }
    items.weights.swap(0, pivot);
    items.parts.swap(0, pivot);

    let (pivot, mut points) = array_unzip_mut(&mut items.points, |p| p.split_at_mut(1));
    let pivot = pivot[coord][0];
    let (_, weights) = items.weights.split_at_mut(1);
    let (_, parts) = items.parts.split_at_mut(1);

    let mut l = 0;
    let mut r = points[0].len();
    loop {
        unsafe {
            let coords = points.get_unchecked(coord);
            while l < r && *coords.get_unchecked(l) < pivot {
                l += 1;
            }
            while l < r && pivot <= *coords.get_unchecked(r - 1) {
                r -= 1;
            }
            if r <= l {
                break;
            }
            r -= 1;
            for p in &mut points {
                p.swap(l, r);
            }
            weights.swap(l, r);
            parts.swap(l, r);
            l += 1;
        }
    }

    for p in &mut items.points {
        p.swap(0, l);
    }
    items.weights.swap(0, l);
    items.parts.swap(0, l);
    let (points_left, points_right) = array_unzip(items.points, |p| p.split_at_mut(l));
    let (weights_left, weights_right) = items.weights.split_at_mut(l);
    let (parts_left, parts_right) = items.parts.split_at_mut(l);
    let left = Items {
        points: points_left,
        weights: weights_left,
        parts: parts_left,
    };
    let right = Items {
        points: points_right,
        weights: weights_right,
        parts: parts_right,
    };
    (left, right)
}

#[cfg(feature = "avx512")]
const AVX2_REGISTER_BYTES: usize = 8;

/// AVX512 (VL+F) version of [reorder_split].
#[cfg(feature = "avx512")]
fn reorder_split_avx512<const D: usize, W>(
    items: Items<'_, D, W>,
    pivot: usize,
    coord: usize,
) -> (Items<'_, D, W>, Items<'_, D, W>) {
    use std::arch::x86_64::__m256;
    use std::arch::x86_64::__m512i;
    use std::arch::x86_64::__mmask8;
    use std::arch::x86_64::_mm256_cmp_ps_mask;
    use std::arch::x86_64::_mm256_loadu_ps;
    use std::arch::x86_64::_mm256_mask_compressstoreu_ps;
    use std::arch::x86_64::_mm256_set1_ps;
    use std::arch::x86_64::_mm512_loadu_epi64;
    use std::arch::x86_64::_mm512_mask_compressstoreu_epi64;
    use std::arch::x86_64::_CMP_LT_OQ;

    let pivot = items.points[coord][pivot];

    unsafe {
        let pivot_val: __m256 = _mm256_set1_ps(pivot);

        let mut lw = 0;
        let mut l = AVX2_REGISTER_BYTES;
        let mut lv: [__m256; D] = array_init(|coord| _mm256_loadu_ps(items.points[coord].as_ptr()));
        let mut lv_weights = _mm512_loadu_epi64(items.weights.as_ptr() as *const i64);
        let mut lv_parts = _mm512_loadu_epi64(items.parts.as_ptr() as *const i64);

        let mut rw = items.points[coord].len();
        let mut r = items.points[coord].len() - AVX2_REGISTER_BYTES;
        let mut rv: [__m256; D] =
            array_init(|coord| _mm256_loadu_ps(items.points[coord].as_ptr().add(r)));
        let mut rv_weights = _mm512_loadu_epi64(items.weights.as_ptr().add(r) as *const i64);
        let mut rv_parts = _mm512_loadu_epi64(items.parts.as_ptr().add(r) as *const i64);

        while l + AVX2_REGISTER_BYTES <= r {
            let free_left = l - lw;
            let free_right = rw - r;

            let val: [__m256; D];
            let val_weights: __m512i;
            let val_parts: __m512i;
            if free_left <= free_right {
                val = lv;
                val_weights = lv_weights;
                val_parts = lv_parts;
                lv = array_init(|coord| _mm256_loadu_ps(items.points[coord].as_ptr().add(l)));
                lv_weights = _mm512_loadu_epi64(items.weights.as_ptr().add(l) as *const i64);
                lv_parts = _mm512_loadu_epi64(items.parts.as_ptr().add(l) as *const i64);
                l += AVX2_REGISTER_BYTES;
            } else {
                val = rv;
                val_weights = rv_weights;
                val_parts = rv_parts;
                r -= AVX2_REGISTER_BYTES;
                rv = array_init(|coord| _mm256_loadu_ps(items.points[coord].as_ptr().add(r)));
                rv_weights = _mm512_loadu_epi64(items.weights.as_ptr().add(r) as *const i64);
                rv_parts = _mm512_loadu_epi64(items.parts.as_ptr().add(r) as *const i64);
            }

            let mask: __mmask8 = _mm256_cmp_ps_mask::<_CMP_LT_OQ>(val[coord], pivot_val);

            let nb_low = mask.count_ones() as usize;
            let nb_high = AVX2_REGISTER_BYTES - nb_low;

            for coord in 0..D {
                _mm256_mask_compressstoreu_ps(
                    items.points[coord].as_mut_ptr().add(lw) as *mut u8,
                    mask,
                    val[coord],
                );
            }
            _mm512_mask_compressstoreu_epi64(
                items.weights.as_mut_ptr().add(lw) as *mut u8,
                mask,
                val_weights,
            );
            _mm512_mask_compressstoreu_epi64(
                items.parts.as_mut_ptr().add(lw) as *mut u8,
                mask,
                val_parts,
            );
            lw += nb_low;

            rw -= nb_high;
            for coord in 0..D {
                _mm256_mask_compressstoreu_ps(
                    items.points[coord].as_mut_ptr().add(rw) as *mut u8,
                    !mask,
                    val[coord],
                );
            }
            _mm512_mask_compressstoreu_epi64(
                items.weights.as_mut_ptr().add(rw) as *mut u8,
                !mask,
                val_weights,
            );
            _mm512_mask_compressstoreu_epi64(
                items.parts.as_mut_ptr().add(rw) as *mut u8,
                !mask,
                val_parts,
            );
        }

        let remaining = r - l;
        let val: [__m256; D] =
            array_init(|coord| _mm256_loadu_ps(items.points[coord].as_ptr().add(l)));
        let val_weights = _mm512_loadu_epi64(items.weights.as_ptr().add(l) as *const i64);
        let val_parts = _mm512_loadu_epi64(items.parts.as_ptr().add(l) as *const i64);
        let mask: __mmask8 = _mm256_cmp_ps_mask::<_CMP_LT_OQ>(val[coord], pivot_val);
        let mask_low = mask & !(0xff << remaining);
        let mask_high = !mask & !(0xff << remaining);
        let nb_low = mask_low.count_ones() as usize;
        let nb_high = mask_high.count_ones() as usize;
        for coord in 0..D {
            _mm256_mask_compressstoreu_ps(
                items.points[coord].as_mut_ptr().add(lw) as *mut u8,
                mask_low,
                val[coord],
            );
        }
        _mm512_mask_compressstoreu_epi64(
            items.weights.as_mut_ptr().add(lw) as *mut u8,
            mask_low,
            val_weights,
        );
        _mm512_mask_compressstoreu_epi64(
            items.parts.as_mut_ptr().add(lw) as *mut u8,
            mask_low,
            val_parts,
        );
        lw += nb_low;
        rw -= nb_high;
        for coord in 0..D {
            _mm256_mask_compressstoreu_ps(
                items.points[coord].as_mut_ptr().add(rw) as *mut u8,
                mask_high,
                val[coord],
            );
        }
        _mm512_mask_compressstoreu_epi64(
            items.weights.as_mut_ptr().add(rw) as *mut u8,
            mask_high,
            val_weights,
        );
        _mm512_mask_compressstoreu_epi64(
            items.parts.as_mut_ptr().add(rw) as *mut u8,
            mask_high,
            val_parts,
        );

        let mask: __mmask8 = _mm256_cmp_ps_mask::<_CMP_LT_OQ>(lv[coord], pivot_val);
        let nb_low = mask.count_ones() as usize;
        let nb_high = AVX2_REGISTER_BYTES - nb_low;
        for coord in 0..D {
            _mm256_mask_compressstoreu_ps(
                items.points[coord].as_mut_ptr().add(lw) as *mut u8,
                mask,
                lv[coord],
            );
        }
        _mm512_mask_compressstoreu_epi64(
            items.weights.as_mut_ptr().add(lw) as *mut u8,
            mask,
            lv_weights,
        );
        _mm512_mask_compressstoreu_epi64(
            items.parts.as_mut_ptr().add(lw) as *mut u8,
            mask,
            lv_parts,
        );
        lw += nb_low;
        rw -= nb_high;
        for coord in 0..D {
            _mm256_mask_compressstoreu_ps(
                items.points[coord].as_mut_ptr().add(rw) as *mut u8,
                !mask,
                lv[coord],
            );
        }
        _mm512_mask_compressstoreu_epi64(
            items.weights.as_mut_ptr().add(rw) as *mut u8,
            !mask,
            lv_weights,
        );
        _mm512_mask_compressstoreu_epi64(
            items.parts.as_mut_ptr().add(rw) as *mut u8,
            !mask,
            lv_parts,
        );

        let mask: __mmask8 = _mm256_cmp_ps_mask::<_CMP_LT_OQ>(rv[coord], pivot_val);
        let nb_low = mask.count_ones() as usize;
        let nb_high = AVX2_REGISTER_BYTES - nb_low;
        for coord in 0..D {
            _mm256_mask_compressstoreu_ps(
                items.points[coord].as_mut_ptr().add(lw) as *mut u8,
                mask,
                rv[coord],
            );
        }
        _mm512_mask_compressstoreu_epi64(
            items.weights.as_mut_ptr().add(lw) as *mut u8,
            mask,
            rv_weights,
        );
        _mm512_mask_compressstoreu_epi64(
            items.parts.as_mut_ptr().add(lw) as *mut u8,
            mask,
            rv_parts,
        );
        lw += nb_low;
        rw -= nb_high;
        for coord in 0..D {
            _mm256_mask_compressstoreu_ps(
                items.points[coord].as_mut_ptr().add(rw) as *mut u8,
                !mask,
                rv[coord],
            );
        }
        _mm512_mask_compressstoreu_epi64(
            items.weights.as_mut_ptr().add(rw) as *mut u8,
            !mask,
            rv_weights,
        );
        _mm512_mask_compressstoreu_epi64(
            items.parts.as_mut_ptr().add(rw) as *mut u8,
            !mask,
            rv_parts,
        );

        let (points_left, points_right) = array_unzip(items.points, |p| p.split_at_mut(lw));
        let (weights_left, weights_right) = items.weights.split_at_mut(lw);
        let (parts_left, parts_right) = items.parts.split_at_mut(lw);
        let left = Items {
            points: points_left,
            weights: weights_left,
            parts: parts_left,
        };
        let right = Items {
            points: points_right,
            weights: weights_right,
            parts: parts_right,
        };
        (left, right)
    }
}

/// Split the arrays in `items` into the ones that have `points[i][coord]`
/// strictly lower than `items.point[pivot][coord]` and the others.
fn reorder_split<const D: usize, W>(
    items: Items<'_, D, W>,
    pivot: usize,
    coord: usize,
) -> (Items<'_, D, W>, Items<'_, D, W>) {
    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vl")
        && std::mem::size_of::<W>() == 8
        && std::mem::size_of::<&AtomicUsize>() == 8
        && items.parts.len() >= 2 * AVX2_REGISTER_BYTES
    {
        return reorder_split_avx512(items, pivot, coord);
    }

    reorder_split_scalar(items, pivot, coord)
}

/// Splits the given items into two sets of similar weights (parallel version).
fn par_rcb_split<const D: usize, W>(
    items: Items<'_, D, W>,
    coord: usize,
    tolerance: f64,
    mut min: f32,
    mut max: f32,
    sum: W,
) -> SplitResult<'_, D, W>
where
    W: RcbWeight,
{
    let span = tracing::info_span!("par_rcb_split");
    let _enter = span.enter();

    let mut prev_count_left = usize::MAX;
    loop {
        let split_target = (min + max) / 2.0;

        // count_left: the number of points that are on the left of split_target
        // weight_left: the weight of all those points
        // nearest_idx: the index in `items` of the point that is 1/ on the
        //    right side of split_target and 2/ the nearest to split_target
        let (count_left, weight_left, nearest_idx, nearest_distance) = items.points[coord]
            .par_iter()
            .with_min_len(4096)
            .zip(&*items.weights)
            .enumerate()
            .fold(
                || (0, W::default(), None, f32::INFINITY),
                |(count, weight_left, mut nearest_idx, mut nearest_distance),
                 (idx, (point, weight))| {
                    let distance = point - split_target;
                    if distance < 0.0 {
                        (
                            count + 1,
                            weight_left + *weight,
                            nearest_idx,
                            nearest_distance,
                        )
                    } else {
                        if distance < nearest_distance {
                            nearest_distance = distance;
                            nearest_idx = Some(idx);
                        }
                        (count, weight_left, nearest_idx, nearest_distance)
                    }
                },
            )
            .reduce(
                || (0, W::default(), None, f32::INFINITY),
                |(count0, weight0, nearest_idx0, nearest_distance0),
                 (count1, weight1, nearest_idx1, nearest_distance1)| {
                    let (nearest_idx, nearest_distance) = if nearest_distance0 < nearest_distance1 {
                        (nearest_idx0, nearest_distance0)
                    } else {
                        (nearest_idx1, nearest_distance1)
                    };
                    (
                        count0 + count1,
                        weight0 + weight1,
                        nearest_idx,
                        nearest_distance,
                    )
                },
            );

        let nearest_idx = match nearest_idx {
            Some(v) => v,
            // Both following cases happen when all points are of the left side
            // of the split_target.  This is the case when min and max are set
            // too loosely, so we let it happen once. If this happens twice,
            // then `prev_count_left` will be the equal to `count_left`.
            None if prev_count_left == count_left => {
                return SplitResult {
                    left: items,
                    right: Items {
                        points: array_init(|_| &mut [][..]),
                        weights: &mut [],
                        parts: &mut [],
                    },
                    weight_left: sum,
                    split_pos: max,
                };
            }
            None => {
                max = split_target;
                prev_count_left = count_left;
                continue;
            }
        };

        let imbalance = {
            let ideal_weight_left = sum.to_f64().unwrap() / 2.0;
            let weight_left = weight_left.to_f64().unwrap();
            f64::abs((weight_left - ideal_weight_left) / ideal_weight_left)
        };
        if count_left == prev_count_left // there is no point between min and max
            || max <= split_target + nearest_distance // or between split_target and max
            || imbalance <= tolerance
        {
            let (left, right) = reorder_split(items, nearest_idx, coord);
            return SplitResult {
                left,
                right,
                weight_left,
                split_pos: split_target,
            };
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

fn rcb_recurse<const D: usize, W>(
    items: Items<'_, D, W>,
    iter_count: usize,
    iter_id: usize,
    coord: usize,
    tolerance: f64,
    sum: W,
    bb: BoundingBox<D>,
) where
    W: RcbWeight,
{
    if items.parts.is_empty() {
        return;
    }
    if iter_count == 0 {
        let span = tracing::info_span!(
            "rcb_recurse: apply_part_id",
            item_count = items.parts.len(),
            iter_id,
        );
        let _enter = span.enter();

        items
            .parts
            .into_par_iter()
            .for_each(|part| part.store(iter_id, Ordering::Relaxed));
        return;
    }

    let min = bb.p_min[coord] as f32;
    let max = bb.p_max[coord] as f32;
    let SplitResult {
        left,
        right,
        weight_left,
        split_pos,
    } = par_rcb_split(items, coord, tolerance, min, max, sum);

    let mut bb_left = bb.clone();
    bb_left.p_max[coord] = split_pos as f64;
    let mut bb_right = bb;
    bb_right.p_min[coord] = split_pos as f64;

    rayon::join(
        || {
            rcb_recurse(
                left,
                iter_count - 1,
                2 * iter_id + 1,
                (coord + 1) % D,
                tolerance,
                weight_left,
                bb_left,
            )
        },
        || {
            rcb_recurse(
                right,
                iter_count - 1,
                2 * iter_id + 2,
                (coord + 1) % D,
                tolerance,
                sum - weight_left,
                bb_right,
            )
        },
    );
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
    P::Iter: rayon::iter::IndexedParallelIterator + Clone,
    W: rayon::iter::IntoParallelIterator,
    W::Item: RcbWeight,
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

    let mut coords = array_init(|coord| {
        points
            .clone()
            .map(|point| point[coord] as f32)
            .collect::<Vec<f32>>()
    });
    let mut weights: Vec<_> = weights.collect();

    let atomic_partition = crate::as_atomic(partition);
    let mut atomic_partition: Vec<&AtomicUsize> = atomic_partition.par_iter().collect();
    let sum = weights.par_iter().cloned().sum();
    let bb = match BoundingBox::from_points(points) {
        Some(v) => v,
        None => return Ok(()), // `items` is empty.
    };

    let points = array_map_mut(&mut coords, |coord| &mut coord[..]);
    let items = Items {
        points,
        weights: &mut weights,
        parts: &mut atomic_partition,
    };
    rcb_recurse(items, iter_count, 0, 0, tolerance, sum, bb);

    // Part IDs must start from zero.
    let part_id_offset = *partition.par_iter().min().unwrap();
    partition
        .par_iter_mut()
        .for_each(|part_id| *part_id -= part_id_offset);

    Ok(())
}

/// Trait alias for values accepted as weights by [Rcb] and [Rib].
pub trait RcbWeight
where
    Self: Copy + std::fmt::Debug + Default + Send + Sync,
    Self: Sum + PartialOrd + ToPrimitive,
    Self: Add<Output = Self> + Sub<Output = Self> + AddAssign,
{
}

impl<T> RcbWeight for T
where
    Self: Copy + std::fmt::Debug + Default + Send + Sync,
    Self: Sum + PartialOrd + ToPrimitive,
    Self: Add<Output = Self> + Sub<Output = Self> + AddAssign,
{
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
/// # fn main() -> Result<(), coupe::Error> {
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
///     .partition(&mut partition, (points, weights))?;
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
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// Berger, M. J. and Bokhari, S. H., 1987. A partitioning strategy for
/// nonuniform problems on multiprocessors. *IEEE Transactions on Computers*,
/// C-36(5):570–580. <doi:10.1109/TC.1987.1676942>.
///
/// Bramas, B., 2017. A Novel Hybrid Quicksort Algorithm Vectorized using
/// AVX-512 on Intel Skylake. *International Journal of Advanced Computer
/// Science and Applications*, 8(10). <doi:10.14569/IJACSA.2017.081044>.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rcb {
    /// The number of iterations of the algorithm. This will yield a partition
    /// of at most `2^num_iter` parts.
    ///
    /// If this equals zero, RCB will not do anything.
    pub iter_count: usize,

    /// Tolerance on the normalized imbalance, for each split.  Please note that
    /// the overall imbalance might end up above this threshold.
    ///
    /// Negative values are interpreted as zeroes.
    pub tolerance: f64,
}

impl<const D: usize, P, W> crate::Partition<(P, W)> for Rcb
where
    P: rayon::iter::IntoParallelIterator<Item = PointND<D>>,
    P::Iter: rayon::iter::IndexedParallelIterator + Clone,
    W: rayon::iter::IntoParallelIterator,
    W::Item: RcbWeight,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (P, W),
    ) -> Result<Self::Metadata, Self::Error> {
        rcb(part_ids, points, weights, self.iter_count, self.tolerance)
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
    tolerance: f64,
) -> Result<(), Error>
where
    Const<D>: DimSub<Const<1>>,
    DefaultAllocator: Allocator<Const<D>, Const<D>, Buffer<f64> = ArrayStorage<f64, D, D>>
        + Allocator<DimDiff<Const<D>, Const<1>>>,
    W: rayon::iter::IntoParallelIterator,
    W::Item: RcbWeight,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    let obb = match OrientedBoundingBox::from_points(points) {
        Some(v) => v,
        None => return Ok(()),
    };
    let points = points.par_iter().map(|p| obb.obb_to_aabb(p));
    // When the rotation is done, we just apply RCB
    rcb(partition, points, weights, n_iter, tolerance)
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
/// # fn main() -> Result<(), coupe::Error> {
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
/// coupe::Rib { iter_count: 1, ..Default::default() }
///     .partition(&mut partition, (&points, weights))?;
///
/// // The two points at the top are in the same part.
/// assert_eq!(partition[0], partition[1]);
///
/// // The two points at the bottom are in the same part.
/// assert_eq!(partition[2], partition[3]);
///
/// // There are two different parts.
/// assert_ne!(partition[1], partition[2]);
/// # Ok(())
/// # }
/// ```
///
/// # Reference
///
/// Williams, Roy D., 1991. Performance of dynamic load balancing algorithms for
/// unstructured mesh calculations. *Concurrency: Practice and Experience*,
/// 3(5):457–481. <doi:10.1002/cpe.4330030502>.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rib {
    /// The number of iterations of the algorithm. This will yield a partition
    /// of at most `2^num_iter` parts.
    pub iter_count: usize,

    /// Same meaning as [`Rcb::tolerance`].
    pub tolerance: f64,
}

impl<'a, const D: usize, W> crate::Partition<(&'a [PointND<D>], W)> for Rib
where
    Const<D>: DimSub<Const<1>> + ToTypenum,
    DefaultAllocator: Allocator<Const<D>, Const<D>, Buffer<f64> = ArrayStorage<f64, D, D>>
        + Allocator<DimDiff<Const<D>, Const<1>>>,
    W: rayon::iter::IntoParallelIterator,
    W::Item: RcbWeight,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (&'a [PointND<D>], W),
    ) -> Result<Self::Metadata, Self::Error> {
        rib(part_ids, points, weights, self.iter_count, self.tolerance)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools as _;
    use proptest::prelude::*;

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

    proptest!(
        #[test]
        fn test_reorder_split_scalar(
            points in (1..24_usize).prop_flat_map(|point_count| {
                prop::collection::vec(prop::num::f32::POSITIVE, point_count)
            }),
        ) {
            for index in 0..points.len() {
                let mut p0 = points.clone();
                let mut weights = vec![1_i32; points.len()];
                let part = AtomicUsize::new(0);
                let mut parts = vec![&part; points.len()];
                let items = Items {
                    points: [&mut p0],
                    weights: &mut weights,
                    parts: &mut parts,
                };
                let pivot = points[index];
                let (left, right) = reorder_split_scalar(items, index, 0);
                prop_assert!(left.points[0].iter().all(|l| *l < pivot));
                prop_assert!(right.points[0].iter().all(|r| pivot <= *r));
                for e in &points {
                    prop_assert_ne!(
                        left.points[0].contains(e),
                        right.points[0].contains(e),
                        "{} is either missing or in both sides",
                        e,
                    );
                }
            }
        }
    );

    #[test]
    #[cfg(feature = "avx512")]
    fn test_reorder_split_simple() {
        const N: usize = 19;
        const SLICE: [f32; N] = [
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0, 0.0,
        ];
        //const SLICE: [f32; N] = [
        //    6.0, 0.0, 3.0, 15.0, 4.0, 5.0, 2.0, 9.0, -1.0, -2.0, 8.0, 12.0, 13.0, -3.0, -4.0, -6.0,
        //    16.0,
        //];
        for index in 0..N {
            let mut p0 = SLICE;
            let mut p1 = [1.0; N];
            let mut weights = [1_i32; N];
            let part = AtomicUsize::new(0);
            let mut parts = [&part; N];
            let items = Items {
                points: [&mut p0, &mut p1],
                weights: &mut weights,
                parts: &mut parts,
            };
            let pivot = SLICE[index];
            let (left, right) = reorder_split_avx512(items, index, 0);
            assert!(left.points[0].iter().all(|l| *l < pivot));
            assert!(right.points[0].iter().all(|r| pivot <= *r));
            for e in SLICE {
                assert!(
                    left.points[0].contains(&e) != right.points[0].contains(&e),
                    "{e} is either missing or in both sides",
                );
            }
        }
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

    proptest!(
        #[test]
        fn test_par_rcb_split(
            (mut points, mut weights) in (1..200_usize)
                .prop_flat_map(|point_count| {
                    (
                        prop::collection::vec(0.0..1.0_f32, point_count),
                        prop::collection::vec(1..1000_u32, point_count),
                    )
                })
        ) {
            let (&min, &max) = points.iter().minmax().into_option().unwrap();
            let sum: u32 = weights.iter().sum();

            let part = AtomicUsize::new(0);
            let mut parts = vec![&part; points.len()];
            let items = Items {
                points: [&mut points],
                weights: &mut weights,
                parts: &mut parts,
            };

            let SplitResult {
                left,
                right,
                weight_left,
                split_pos,
            } = par_rcb_split(items, 0, 0.05, min, max, sum);

            prop_assert!(left.points[0].iter().all(|l| *l < split_pos));
            prop_assert!(right.points[0].iter().all(|r| split_pos <= *r));
            prop_assert_eq!(weight_left, left.weights.iter().sum());
            prop_assert_eq!(sum - weight_left, right.weights.iter().sum());
        }
    );

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
}
