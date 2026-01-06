#![allow(clippy::missing_safety_doc)] // See include/coupe.h

use crate::data::Data;
use crate::data::Type;
use coupe::nalgebra::allocator::Allocator;
use coupe::nalgebra::ArrayStorage;
use coupe::nalgebra::Const;
use coupe::nalgebra::DefaultAllocator;
use coupe::nalgebra::DimDiff;
use coupe::nalgebra::DimSub;
use coupe::nalgebra::ToTypenum;
use coupe::sprs::CsMatView;
use coupe::sprs::CSR;
use coupe::Partition as _;
use coupe::Point2D;
use coupe::PointND;
use coupe::Real;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::os::raw::c_int;
use std::ptr;
use std::slice;

#[macro_use]
mod data;

fn box_try_new<T>(value: T) -> Option<Box<T>>
where
    T: std::panic::UnwindSafe,
{
    // TODO replace with [Box::try_new] once stabilised.
    std::panic::catch_unwind(|| Some(Box::new(value))).unwrap_or(None)
}

#[repr(C)]
pub enum Error {
    Ok,
    Alloc,
    Crash,
    BadDimension,
    BadType,
    BipartOnly,
    LenMismatch,
    NotFound,
    NegValues,
}

impl From<coupe::Error> for Error {
    fn from(err: coupe::Error) -> Self {
        match err {
            coupe::Error::NotFound => Self::NotFound,
            coupe::Error::NegativeValues => Self::NegValues,
            coupe::Error::BiPartitioningOnly => Self::BipartOnly,
            coupe::Error::InputLenMismatch { .. } => Self::LenMismatch,
            _ => unreachable!(),
        }
    }
}

/// Wrapper around [std::panic::catch_unwind] that returns [Error::Crash] on
/// panic.
fn catch_unwind<F>(f: F) -> Error
where
    F: FnOnce() -> Error + std::panic::UnwindSafe,
{
    std::panic::catch_unwind(f).unwrap_or(Error::Crash)
}

#[no_mangle]
pub extern "C" fn coupe_strerror(err: Error) -> *const c_char {
    match err {
        Error::Ok => {
            static MSG: &[u8] = b"success\0";
            MSG.as_ptr() as *const c_char
        }
        Error::Alloc => {
            static MSG: &[u8] = b"allocation failed\0";
            MSG.as_ptr() as *const c_char
        }
        Error::Crash => {
            static MSG: &[u8] = b"coupe encountered a bug and crashed\0";
            MSG.as_ptr() as *const c_char
        }
        Error::BadDimension => {
            static MSG: &[u8] = b"this algorithm does not support the given mesh dimension\0";
            MSG.as_ptr() as *const c_char
        }
        Error::BadType => {
            static MSG: &[u8] = b"this algorithm does not support the given type\0";
            MSG.as_ptr() as *const c_char
        }
        Error::BipartOnly => {
            static MSG: &[u8] = b"this algorithm does not support k-way partitioning\0";
            MSG.as_ptr() as *const c_char
        }
        Error::LenMismatch => {
            static MSG: &[u8] =
                b"input iters (e.g. weights and points) don't have the same length\0";
            MSG.as_ptr() as *const c_char
        }
        Error::NotFound => {
            static MSG: &[u8] = b"no partition has been found for the given constraints\0";
            MSG.as_ptr() as *const c_char
        }
        Error::NegValues => {
            static MSG: &[u8] = b"this algorithm does not support negative values\0";
            MSG.as_ptr() as *const c_char
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn coupe_data_free(data: *mut Data) {
    if !data.is_null() {
        let data = Box::from_raw(data);
        drop(data);
    }
}

#[no_mangle]
pub extern "C" fn coupe_data_array(len: usize, type_: Type, array: *const c_void) -> *mut Data {
    let data = Data::Array(data::Array { len, type_, array });
    match box_try_new(data) {
        Some(v) => Box::into_raw(v),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn coupe_data_constant(len: usize, type_: Type, value: *const c_void) -> *mut Data {
    let data = Data::Constant(data::Constant { len, type_, value });
    match box_try_new(data) {
        Some(v) => Box::into_raw(v),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn coupe_data_fn(
    context: *const c_void,
    len: usize,
    type_: Type,
    i_th: extern "C" fn(*const c_void, usize) -> *const c_void,
) -> *mut Data {
    let data = Data::Fn(data::Fn {
        len,
        type_,
        context,
        i_th,
    });
    match box_try_new(data) {
        Some(v) => Box::into_raw(v),
        None => ptr::null_mut(),
    }
}

pub enum Adjncy<'a> {
    Int(CsMatView<'a, c_int>),
    Int64(CsMatView<'a, i64>),
    Double(CsMatView<'a, f64>),
}

#[no_mangle]
pub unsafe extern "C" fn coupe_adjncy_free(adjncy: *mut Adjncy) {
    if !adjncy.is_null() {
        let adjncy = Box::from_raw(adjncy);
        drop(adjncy);
    }
}

unsafe fn adjncy_csr_unchecked(
    size: usize,
    xadj: *const usize,
    adjncy: *const usize,
    data_type: Type,
    data: *const c_void,
) -> Adjncy<'static> {
    let xadj = slice::from_raw_parts(xadj, size + 1);
    let adjncy = slice::from_raw_parts(adjncy, xadj[xadj.len() - 1]);
    match data_type {
        Type::Int => {
            let data = slice::from_raw_parts(data as *const c_int, adjncy.len());
            let matrix = CsMatView::new_unchecked(CSR, (size, size), xadj, adjncy, data);
            Adjncy::Int(matrix)
        }
        Type::Int64 => {
            let data = slice::from_raw_parts(data as *const i64, adjncy.len());
            let matrix = CsMatView::new_unchecked(CSR, (size, size), xadj, adjncy, data);
            Adjncy::Int64(matrix)
        }
        Type::Double => {
            let data = slice::from_raw_parts(data as *const f64, adjncy.len());
            let matrix = CsMatView::new_unchecked(CSR, (size, size), xadj, adjncy, data);
            Adjncy::Double(matrix)
        }
    }
}

fn adjncy_ptr(adjacency: Adjncy<'_>) -> *mut Adjncy<'_> {
    match box_try_new(adjacency) {
        Some(v) => Box::into_raw(v),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn coupe_adjncy_csr(
    size: usize,
    xadj: *const usize,
    adjncy: *const usize,
    data_type: Type,
    data: *const c_void,
) -> *mut Adjncy<'static> {
    let adjacency = adjncy_csr_unchecked(size, xadj, adjncy, data_type, data);
    match adjacency {
        Adjncy::Int(matrix) => {
            if matrix.check_compressed_structure().is_err() {
                return ptr::null_mut();
            }
        }
        Adjncy::Int64(matrix) => {
            if matrix.check_compressed_structure().is_err() {
                return ptr::null_mut();
            }
        }
        Adjncy::Double(matrix) => {
            if matrix.check_compressed_structure().is_err() {
                return ptr::null_mut();
            }
        }
    }
    adjncy_ptr(adjacency)
}

#[no_mangle]
pub unsafe extern "C" fn coupe_adjncy_csr_unchecked(
    size: usize,
    xadj: *const usize,
    adjncy: *const usize,
    data_type: Type,
    data: *const c_void,
) -> *mut Adjncy<'static> {
    let adjacency = adjncy_csr_unchecked(size, xadj, adjncy, data_type, data);
    adjncy_ptr(adjacency)
}

unsafe fn coupe_rcb_d<const D: usize>(
    partition: &mut [usize],
    points: &Data,
    weights: &Data,
    mut algo: coupe::Rcb,
) -> Error {
    let res = with_par_iter!(points, PointND<D>, {
        with_par_iter!(weights, { algo.partition(partition, (points, weights)) })
    });
    match res {
        Ok(_) => Error::Ok,
        Err(err) => Error::from(err),
    }
}

#[no_mangle]
pub unsafe extern "C" fn coupe_rcb(
    partition: *mut usize,
    dimension: usize,
    points: *const Data,
    weights: *const Data,
    iter_count: usize,
    tolerance: f64,
) -> Error {
    let points = &*points;
    let weights = &*weights;

    let element_count = points.len();
    if element_count != weights.len() {
        return Error::LenMismatch;
    }

    let algo = coupe::Rcb {
        iter_count,
        tolerance,
    };

    catch_unwind(|| {
        let partition = slice::from_raw_parts_mut(partition, element_count);
        match dimension {
            2 => coupe_rcb_d::<2>(partition, points, weights, algo),
            3 => coupe_rcb_d::<3>(partition, points, weights, algo),
            _ => Error::BadDimension,
        }
    })
}

unsafe fn coupe_rib_d<const D: usize>(
    partition: &mut [usize],
    points: &Data,
    weights: &Data,
    mut algo: coupe::Rib,
) -> Error
where
    Const<D>: DimSub<Const<1>> + ToTypenum,
    DefaultAllocator: Allocator<Const<D>, Const<D>, Buffer<f64> = ArrayStorage<f64, D, D>>
        + Allocator<DimDiff<Const<D>, Const<1>>>,
{
    let points = match points.to_slice::<PointND<D>>() {
        Ok(v) => v,
        Err(_) => return Error::Alloc,
    };
    let res = with_par_iter!(weights, {
        algo.partition(partition, (points.as_ref(), weights))
    });
    match res {
        Ok(_) => Error::Ok,
        Err(err) => Error::from(err),
    }
}

#[no_mangle]
pub unsafe extern "C" fn coupe_rib(
    partition: *mut usize,
    dimension: usize,
    points: *const Data,
    weights: *const Data,
    iter_count: usize,
    tolerance: f64,
) -> Error {
    let points = &*points;
    let weights = &*weights;

    let element_count = points.len();
    if element_count != weights.len() {
        return Error::LenMismatch;
    }

    let algo = coupe::Rib {
        iter_count,
        tolerance,
    };

    catch_unwind(|| {
        let partition = slice::from_raw_parts_mut(partition, element_count);
        match dimension {
            2 => coupe_rib_d::<2>(partition, points, weights, algo),
            3 => coupe_rib_d::<3>(partition, points, weights, algo),
            _ => Error::BadDimension,
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn coupe_hilbert(
    partition: *mut usize,
    points: *const Data,
    weights: *const Data,
    part_count: usize,
    order: u32,
) -> Error {
    let points = &*points;
    let weights = &*weights;

    let element_count = points.len();
    if element_count != weights.len() {
        return Error::LenMismatch;
    }
    if weights.type_() != Type::Double {
        return Error::BadType;
    }

    catch_unwind(|| {
        let partition = slice::from_raw_parts_mut(partition, element_count);

        let points = match points.to_slice::<Point2D>() {
            Ok(v) => v,
            Err(_) => return Error::Alloc,
        };
        let weights = match weights.to_slice::<f64>() {
            Ok(v) => v,
            Err(_) => return Error::Alloc,
        };

        let res =
            coupe::HilbertCurve { part_count, order }.partition(partition, (&*points, weights));

        match res {
            Ok(()) => Error::Ok,
            Err(_) => Error::NotFound, // TODO use a proper error code
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn coupe_greedy(
    partition: *mut usize,
    weights: *const Data,
    part_count: usize,
) -> Error {
    let weights = &*weights;
    let element_count = weights.len();
    catch_unwind(|| {
        let partition = slice::from_raw_parts_mut(partition, element_count);
        let mut algo = coupe::Greedy { part_count };
        match with_iter!(weights, { algo.partition(partition, weights) }) {
            Ok(()) => Error::Ok,
            Err(err) => Error::from(err),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn coupe_karmarkar_karp(
    partition: *mut usize,
    weights: *const Data,
    part_count: usize,
) -> Error {
    let weights = &*weights;
    let element_count = weights.len();

    catch_unwind(|| {
        let partition = slice::from_raw_parts_mut(partition, element_count);

        let mut algo = coupe::KarmarkarKarp { part_count };

        let res = match weights.type_() {
            Type::Int => {
                with_iter!(weights, c_int, { algo.partition(partition, weights) })
            }
            Type::Int64 => {
                with_iter!(weights, i64, { algo.partition(partition, weights) })
            }
            Type::Double => {
                with_iter!(weights, Real, { algo.partition(partition, weights) })
            }
        };

        match res {
            Ok(()) => Error::Ok,
            Err(err) => Error::from(err),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn coupe_karmarkar_karp_complete(
    partition: *mut usize,
    weights: *const Data,
    tolerance: f64,
) -> Error {
    let weights = &*weights;
    let element_count = weights.len();
    catch_unwind(|| {
        let partition = slice::from_raw_parts_mut(partition, element_count);
        let mut algo = coupe::CompleteKarmarkarKarp { tolerance };
        match with_iter!(weights, { algo.partition(partition, weights) }) {
            Ok(()) => Error::Ok,
            Err(err) => Error::from(err),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn coupe_fiduccia_mattheyses(
    partition: *mut usize,
    adjncy: *const Adjncy<'_>,
    weights: *const Data,
    max_passes: usize,
    max_moves_per_pass: usize,
    max_imbalance: f64,
    max_bad_moves_in_a_row: usize,
) -> Error {
    let weights = &*weights;
    let element_count = weights.len();

    let mut algo = coupe::FiducciaMattheyses {
        max_passes: if max_passes == 0 {
            None
        } else {
            Some(max_passes)
        },
        max_moves_per_pass: if max_moves_per_pass == 0 {
            None
        } else {
            Some(max_moves_per_pass)
        },
        max_imbalance: if max_imbalance <= 0.0 {
            None
        } else {
            Some(max_imbalance)
        },
        max_bad_move_in_a_row: max_bad_moves_in_a_row,
    };

    let adjacency = match &*adjncy {
        Adjncy::Int64(matrix) => *matrix,
        _ => return Error::BadType,
    };

    catch_unwind(move || {
        let partition = slice::from_raw_parts_mut(partition, element_count);

        let res = with_slice!(weights, {
            algo.partition(partition, (adjacency, weights.as_ref()))
        });

        match res {
            Ok(_) => Error::Ok, // TODO use metadata
            Err(err) => Error::from(err),
        }
    })
}
