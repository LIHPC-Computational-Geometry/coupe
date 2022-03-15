#![allow(clippy::missing_safety_doc)] // See `meshio.h`.

use libc::c_int;
use std::fs;
use std::io;
use std::mem;
use std::os::unix::io::FromRawFd as _;
use std::os::unix::io::IntoRawFd as _;
use std::ptr;

const ERROR_OTHER: c_int = -1;
const ERROR_BAD_HEADER: c_int = -2;
const ERROR_UNSUPPORTED_VERSION: c_int = -3;

fn err_partition_code(err: mesh_io::partition::Error) -> c_int {
    match err {
        mesh_io::partition::Error::BadHeader => ERROR_BAD_HEADER,
        mesh_io::partition::Error::UnsupportedVersion => ERROR_UNSUPPORTED_VERSION,
        mesh_io::partition::Error::Io(_) => ERROR_OTHER,
    }
}

#[no_mangle]
pub unsafe extern "C" fn mio_partition_read(
    size: *mut u64,
    partition: *mut *mut u64,
    fd: c_int,
) -> c_int {
    let f = fs::File::from_raw_fd(fd);
    let r = io::BufReader::new(f);
    match mesh_io::partition::read(r) {
        Ok(p) => {
            let mut p = p.into_boxed_slice();
            *size = p.len() as u64;
            *partition = p.as_mut_ptr() as *mut u64;
            0
        }
        Err(err) => err_partition_code(err),
    }
}

#[no_mangle]
pub unsafe extern "C" fn mio_partition_write(fd: c_int, size: u64, partition: *const u64) -> c_int {
    let f = fs::File::from_raw_fd(fd);
    let mut w = io::BufWriter::new(f);
    let p = std::slice::from_raw_parts(partition, size as usize)
        .iter()
        .map(|id| *id as usize);
    if mesh_io::partition::write(&mut w, p).is_err() {
        return ERROR_OTHER;
    }
    match w.into_inner() {
        Ok(f) => {
            f.into_raw_fd();
            0
        }
        Err(_) => ERROR_OTHER,
    }
}

#[no_mangle]
pub unsafe extern "C" fn mio_partition_free(size: u64, partition: *mut u64) {
    let size = size as usize;
    mem::drop(Vec::from_raw_parts(partition, size, size));
}

#[no_mangle]
pub unsafe extern "C" fn mio_weights_read(fd: c_int) -> *mut mesh_io::weight::Array {
    let f = fs::File::from_raw_fd(fd);
    let mut r = io::BufReader::new(f);
    let w = match mesh_io::weight::read(&mut r) {
        Ok(w) => w,
        Err(_) => return ptr::null_mut(),
    };
    Box::into_raw(Box::new(w))
}

#[no_mangle]
pub unsafe extern "C" fn mio_weights_count(weights: *mut mesh_io::weight::Array) -> u64 {
    assert!(!weights.is_null());

    let weights = Box::from_raw(weights);
    let len = match weights.as_ref() {
        mesh_io::weight::Array::Integers(is) => is.len(),
        mesh_io::weight::Array::Floats(fs) => fs.len(),
    };
    mem::forget(weights);

    len.try_into().unwrap()
}

#[no_mangle]
pub unsafe extern "C" fn mio_weights_first_criterion(
    criterion: *mut f64,
    weights: *mut mesh_io::weight::Array,
) {
    assert!(!weights.is_null());

    let weights = Box::from_raw(weights);
    let len = match weights.as_ref() {
        mesh_io::weight::Array::Integers(is) => is.len(),
        mesh_io::weight::Array::Floats(fs) => fs.len(),
    };
    let criterion = std::slice::from_raw_parts_mut(criterion, len);

    match weights.as_ref() {
        mesh_io::weight::Array::Integers(is) => {
            for (c, w) in criterion.iter_mut().zip(is) {
                *c = w[0] as f64;
            }
        }
        mesh_io::weight::Array::Floats(fs) => {
            for (c, w) in criterion.iter_mut().zip(fs) {
                *c = w[0];
            }
        }
    }

    mem::forget(weights);
}

#[no_mangle]
pub unsafe extern "C" fn mio_weights_free(weights: *mut mesh_io::weight::Array) {
    if !weights.is_null() {
        mem::drop(Box::from_raw(weights));
    }
}

#[no_mangle]
pub unsafe extern "C" fn mio_medit_read(fd: c_int) -> *mut mesh_io::medit::Mesh {
    let f = fs::File::from_raw_fd(fd);
    let mut r = io::BufReader::new(f);
    let m = match mesh_io::medit::Mesh::from_reader(&mut r) {
        Ok(m) => m,
        Err(_) => return ptr::null_mut(),
    };
    Box::into_raw(Box::new(m))
}

#[no_mangle]
pub unsafe extern "C" fn mio_medit_free(medit: *mut mesh_io::medit::Mesh) {
    if !medit.is_null() {
        mem::drop(Box::from_raw(medit));
    }
}

#[no_mangle]
pub unsafe extern "C" fn mio_medit_dimension(medit: *mut mesh_io::medit::Mesh) -> c_int {
    assert!(!medit.is_null());

    let medit = Box::from_raw(medit);
    let dimension = medit.dimension();
    mem::forget(medit);

    dimension.try_into().unwrap()
}

#[no_mangle]
pub unsafe extern "C" fn mio_medit_node_count(medit: *mut mesh_io::medit::Mesh) -> u64 {
    assert!(!medit.is_null());

    let medit = Box::from_raw(medit);
    let count = medit.node_count();
    mem::forget(medit);

    count.try_into().unwrap()
}

#[no_mangle]
pub unsafe extern "C" fn mio_medit_coordinates(
    medit: *mut mesh_io::medit::Mesh,
    node_idx: usize,
) -> *const f64 {
    assert!(!medit.is_null());

    let medit = Box::from_raw(medit);
    let node = medit.node(node_idx).as_ptr();
    mem::forget(medit);

    node
}

#[no_mangle]
pub unsafe extern "C" fn mio_medit_element_count(medit: *mut mesh_io::medit::Mesh) -> u64 {
    assert!(!medit.is_null());

    let medit = Box::from_raw(medit);
    let count = medit.element_count();
    mem::forget(medit);

    count.try_into().unwrap()
}

#[repr(C)]
pub struct MeditElement {
    dimension: c_int,
    node_count: c_int,
    nodes: *const usize,
}

#[no_mangle]
pub unsafe extern "C" fn mio_medit_element(
    element: *mut MeditElement,
    medit: *mut mesh_io::medit::Mesh,
    element_idx: usize,
) {
    assert!(!medit.is_null());

    let medit = Box::from_raw(medit);
    if let Some((el_type, el_nodes, _el_ref)) = medit.elements().nth(element_idx) {
        *element = MeditElement {
            dimension: el_type.dimension().try_into().unwrap(),
            node_count: el_nodes.len().try_into().unwrap(),
            nodes: el_nodes.as_ptr(),
        };
    }
    mem::forget(medit);
}
