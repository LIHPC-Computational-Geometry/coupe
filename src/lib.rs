#[cfg(test)]
#[macro_use]
extern crate approx;
#[cfg(not(test))]
extern crate approx;
extern crate itertools;
extern crate nalgebra;
extern crate rayon;
extern crate snowflake;

pub mod algorithms;
pub mod analysis;
pub mod geometry;

#[cfg(test)]
mod tests;

// API

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::{DimDiff, DimSub};
use nalgebra::DefaultAllocator;
use nalgebra::DimName;
use nalgebra::U1;

use snowflake::ProcessUniqueId;

use std::marker::PhantomData;

use crate::geometry::PointND;

pub trait InitialPartition<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    fn partition(&self, points: &[PointND<D>], weights: &[f64]) -> Vec<ProcessUniqueId>;
}

pub trait ImprovePartition<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    fn improve_partition(
        &self,
        points: &[PointND<D>],
        weights: &[f64],
        partition: &mut [ProcessUniqueId],
    );
}

pub struct Rcb {
    pub num_iter: usize,
}

impl<D> InitialPartition<D> for Rcb
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
    fn partition(&self, points: &[PointND<D>], weights: &[f64]) -> Vec<ProcessUniqueId> {
        crate::algorithms::recursive_bisection::rcb(points, weights, self.num_iter)
    }
}

pub struct ZCurve {
    pub num_partitions: usize,
    pub order: u32,
}

impl<D> InitialPartition<D> for ZCurve
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    fn partition(&self, points: &[PointND<D>], _weights: &[f64]) -> Vec<ProcessUniqueId> {
        crate::algorithms::z_curve::z_curve_partition(points, self.num_partitions, self.order)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KMeans {
    pub num_partitions: usize,
    pub imbalance_tol: f64,
    pub delta_threshold: f64,
    pub max_iter: usize,
    pub max_balance_iter: usize,
    pub erode: bool,
    pub hilbert: bool,
    pub mbr_early_break: bool,
}

impl Default for KMeans {
    fn default() -> Self {
        Self {
            num_partitions: 7,
            imbalance_tol: 5.,
            delta_threshold: 0.01,
            max_iter: 500,
            max_balance_iter: 20, // for now, `max_balance_iter > 1` yields poor convergence time
            erode: false,         // for now, `erode` yields` enabled yields wrong results
            hilbert: true,
            mbr_early_break: false, // for now, `mbr_early_break` enabled yields wrong results
        }
    }
}

impl<D> ImprovePartition<D> for KMeans
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    fn improve_partition(
        &self,
        points: &[PointND<D>],
        weights: &[f64],
        partition: &mut [ProcessUniqueId],
    ) {
        let settings = crate::algorithms::k_means::BalancedKmeansSettings {
            num_partitions: self.num_partitions,
            imbalance_tol: self.imbalance_tol,
            delta_threshold: self.delta_threshold,
            max_iter: self.max_iter,
            max_balance_iter: self.max_balance_iter,
            erode: self.erode,
            hilbert: self.hilbert,
            mbr_early_break: self.mbr_early_break,
        };
        crate::algorithms::k_means::balanced_k_means_with_initial_partition(
            points, weights, settings, partition,
        )
    }
}

struct ImproverComposition<D, T, U>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    T: ImprovePartition<D>,
    U: ImprovePartition<D>,
{
    first: T,
    second: U,
    _marker: PhantomData<D>,
}

struct InitialImproverComposition<D, T, U>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    T: InitialPartition<D>,
    U: ImprovePartition<D>,
{
    first: T,
    second: U,
    _marker: PhantomData<D>,
}

impl<D, T, U> ImprovePartition<D> for ImproverComposition<D, T, U>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    T: ImprovePartition<D>,
    U: ImprovePartition<D>,
{
    fn improve_partition(
        &self,
        points: &[PointND<D>],
        weights: &[f64],
        partition: &mut [ProcessUniqueId],
    ) {
        self.first.improve_partition(points, weights, partition);
        self.second.improve_partition(points, weights, partition);
    }
}

impl<D, T, U> InitialPartition<D> for InitialImproverComposition<D, T, U>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    T: InitialPartition<D>,
    U: ImprovePartition<D>,
{
    fn partition(&self, points: &[PointND<D>], weights: &[f64]) -> Vec<ProcessUniqueId> {
        let mut partition = self.first.partition(points, weights);
        self.second
            .improve_partition(points, weights, &mut partition);
        partition
    }
}

pub fn compose_two_improvers<D>(
    improver_1: impl ImprovePartition<D>,
    improver_2: impl ImprovePartition<D>,
) -> impl ImprovePartition<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    ImproverComposition {
        first: improver_1,
        second: improver_2,
        _marker: PhantomData,
    }
}

pub fn compose_two_initial_improver<D>(
    initial: impl InitialPartition<D>,
    improver: impl ImprovePartition<D>,
) -> impl InitialPartition<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    InitialImproverComposition {
        first: initial,
        second: improver,
        _marker: PhantomData,
    }
}
