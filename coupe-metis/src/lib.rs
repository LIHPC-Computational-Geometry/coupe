#![allow(clippy::missing_safety_doc)] // See the MeTiS manual

use coupe::Partition as _;
use std::slice;

pub type Idx = i32;
pub type Real = f32;

//const NOPTIONS: usize = 40; // TODO

#[repr(C)]
pub enum RStatus {
    Ok = 1,
    ErrorInput = -2,
    ErrorMemory = -3,
    Error = -4,
}

#[allow(clippy::too_many_arguments)]
unsafe fn partition(
    nvtxs: *const Idx,
    ncon: *const Idx,
    xadj: *const Idx,
    adjncy: *const Idx,
    vwgt: *const Idx,
    _vsize: *const Idx, // TODO
    adjwgt: *const Idx,
    nparts: *const Idx,
    _tpwgts: *const Idx,  // TODO
    _ubvec: *const Real,  // TODO
    _options: *const Idx, // TODO
    objval: *mut Idx,
    part: *mut Idx,
) -> RStatus {
    let nvtxs = *nvtxs as usize;
    let ncon = *ncon as usize;
    if ncon != 1 {
        // TODO make graph algorithms generic over input collection.
        return RStatus::Error;
    }
    let nparts = *nparts as usize;

    // TODO make graph algorithms work over CsMatViewI instead of CsMatView
    // to avoid these conversions.
    let xadj: Vec<usize> = slice::from_raw_parts(xadj, nvtxs + 1)
        .iter()
        .map(|i| *i as usize)
        .collect();
    let adjncy: Vec<usize> = slice::from_raw_parts(adjncy, xadj[xadj.len() - 1] as usize)
        .iter()
        .map(|i| *i as usize)
        .collect();
    let adjwgt = if adjwgt.is_null() {
        vec![1_i64; adjncy.len()]
    } else {
        // TODO make graph algorithms generic over weight type
        slice::from_raw_parts(adjwgt, adjncy.len())
            .iter()
            .map(|wgt| *wgt as i64)
            .collect()
    };
    let adjacency = sprs::CsMat::new((nvtxs, nvtxs), xadj, adjncy, adjwgt);

    let mut partition = vec![0; nvtxs];

    let vwgt = if vwgt.is_null() {
        vec![0.0; nvtxs]
    } else {
        // TODO make graph algorithms generic over weight type
        slice::from_raw_parts(vwgt, nvtxs)
            .iter()
            .map(|wgt| *wgt as f64)
            .collect()
    };

    let ret = coupe::KarmarkarKarp { part_count: nparts }
        .partition(&mut partition, vwgt.iter().map(|w| coupe::Real::from(*w)));
    if let Err(err) = ret {
        println!("error: {}", err);
        return RStatus::Error;
    }

    let ret =
        coupe::FiducciaMattheyses::default().partition(&mut partition, (adjacency.view(), &vwgt));
    if let Err(err) = ret {
        println!("error: {}", err);
        return RStatus::Error;
    }

    let part = slice::from_raw_parts_mut(part, nvtxs);
    for (dst, src) in part.iter_mut().zip(&mut partition) {
        *dst = *src as Idx;
    }

    // TODO have fidducia return the cut size.
    *objval = 0;

    RStatus::Ok
}

#[no_mangle]
pub unsafe extern "C" fn METIS_PartGraphKway(
    nvtxs: *const Idx,
    ncon: *const Idx,
    xadj: *const Idx,
    adjncy: *const Idx,
    vwgt: *const Idx,
    vsize: *const Idx,
    adjwgt: *const Idx,
    nparts: *const Idx,
    tpwgts: *const Idx,
    ubvec: *const Real,
    options: *const Idx,
    objval: *mut Idx,
    part: *mut Idx,
) -> RStatus {
    partition(
        nvtxs, ncon, xadj, adjncy, vwgt, vsize, adjwgt, nparts, tpwgts, ubvec, options, objval,
        part,
    )
}

#[no_mangle]
pub unsafe extern "C" fn METIS_PartGraphRecursive(
    nvtxs: *const Idx,
    ncon: *const Idx,
    xadj: *const Idx,
    adjncy: *const Idx,
    vwgt: *const Idx,
    vsize: *const Idx,
    adjwgt: *const Idx,
    nparts: *const Idx,
    tpwgts: *const Idx,
    ubvec: *const Real,
    options: *const Idx,
    objval: *mut Idx,
    part: *mut Idx,
) -> RStatus {
    partition(
        nvtxs, ncon, xadj, adjncy, vwgt, vsize, adjwgt, nparts, tpwgts, ubvec, options, objval,
        part,
    )
}
