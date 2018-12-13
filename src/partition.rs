//! Utilities to manipulate partitions

use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::{DimDiff, DimSub};
use nalgebra::DefaultAllocator;
use nalgebra::DimName;
use nalgebra::U1;

use num::{Num, Signed};
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::PartialOrd;
use std::iter::Sum;

use crate::geometry::Mbr;
use crate::PointND;

pub struct Partition<'a, P, W> {
    points: &'a [P],
    weights: &'a [W],
    ids: Vec<ProcessUniqueId>,
}

impl<'a, P, W> Partition<'a, P, W> {
    // panics if points and weights have different sizes
    pub fn new(points: &'a [P], weights: &'a [W]) -> Self {
        if points.len() != weights.len() {
            panic!(
                "Cannot initialize a partition with points and weights of different sizes. Found {} points and {} weights",
                points.len(), weights.len()
            );
        }

        // allocate a new vector
        let dummy_id = ProcessUniqueId::new();
        let ids = (0..points.len())
            .into_par_iter()
            .map(|_| dummy_id)
            .collect();

        Self {
            points,
            weights,
            ids,
        }
    }

    pub fn from_ids(points: &'a [P], weights: &'a [W], ids: Vec<ProcessUniqueId>) -> Self {
        if points.len() != weights.len() {
            panic!(
                "Cannot initialize a partition with points and weights of different sizes. Found {} points and {} weights",
                points.len(), weights.len()
            );
        }

        if points.len() != ids.len() {
            panic!(
                "Cannot initialize a partition with points and ids of different sizes. Found {} points and {} ids",
                points.len(), ids.len()
            );
        }

        Self {
            points,
            weights,
            ids,
        }
    }

    /// Returns the array of ids representing the partition,
    /// consuming the partition
    pub fn into_ids(self) -> Vec<ProcessUniqueId> {
        self.ids
    }

    pub fn ids(&self) -> &[ProcessUniqueId] {
        &self.ids
    }

    pub fn points(&self) -> &[P] {
        &self.points
    }

    pub fn weights(&self) -> &[W] {
        &self.weights
    }

    pub fn parts(&'a self) -> impl Iterator<Item = Part<'a, P, W>> {
        let indices = (0..self.points.len()).collect::<Vec<_>>();
        self.ids.iter().unique().map(move |id| {
            let indices = indices
                .iter()
                .filter(|idx| self.ids[**idx] == *id)
                .cloned()
                .collect::<Vec<_>>();
            Part::<'a, P, W> {
                partition: &self,
                indices,
            }
        })
    }

    pub fn max_imbalance(&'a self) -> W
    where
        W: Num + PartialOrd + Signed + Sum + Clone,
    {
        let total_weights = self
            .parts()
            .map(|part| part.total_weight())
            .collect::<Vec<_>>();
        total_weights
            .iter()
            .flat_map(|w1| {
                total_weights
                    .iter()
                    .map(move |w2| (w1.clone() - w2.clone()).abs())
            })
            .max_by(|a, b| a.partial_cmp(&b).unwrap())
            // if the partition is empty, then there is the imbalance is null
            .unwrap_or(W::zero())
    }

    pub fn relative_imbalance(&'a self) -> W
    where
        W: Num + PartialOrd + Signed + Sum + Clone,
    {
        let total_weights = self
            .parts()
            .map(|part| part.total_weight())
            .collect::<Vec<_>>();
        let max_imbalance = total_weights
            .iter()
            .flat_map(|w1| {
                total_weights
                    .iter()
                    .map(move |w2| (w1.clone() - w2.clone()).abs())
            })
            .max_by(|a, b| a.partial_cmp(&b).unwrap())
            .unwrap_or(W::zero());

        max_imbalance / total_weights.into_iter().sum()
    }
}

pub struct Part<'a, P, W> {
    partition: &'a Partition<'a, P, W>,
    indices: Vec<usize>,
}

impl<'a, P, W> Part<'a, P, W> {
    pub fn total_weight(&self) -> W
    where
        W: Sum + Clone,
    {
        self.indices
            .iter()
            .map(|idx| &self.partition.weights[*idx])
            .cloned()
            .sum()
    }

    pub fn iter(&self) -> PartIter<P, W> {
        PartIter {
            partition: self.partition,
            indices: &self.indices,
        }
    }
}

impl<'a, W, D> Part<'a, PointND<D>, W>
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D>
        + Allocator<f64, D, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    pub fn aspect_ratio(&self) -> f64 {
        Mbr::from_points(self.partition.points()).aspect_ratio()
    }
}

pub struct PartIter<'a, P, W> {
    partition: &'a Partition<'a, P, W>,
    indices: &'a [usize],
}

impl<'a, P, W> Iterator for PartIter<'a, P, W> {
    type Item = (&'a P, &'a W);
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.get(0).and_then(|idx| {
            self.indices = &self.indices[1..];
            Some((&self.partition.points[*idx], &self.partition.weights[*idx]))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Point2D;

    #[test]
    fn test_partition() {
        let points = vec![
            Point2D::new(0., 0.),
            Point2D::new(1., 0.),
            Point2D::new(2., 0.),
            Point2D::new(3., 0.),
            Point2D::new(4., 0.),
            Point2D::new(5., 0.),
            Point2D::new(6., 0.),
            Point2D::new(7., 0.),
            Point2D::new(8., 0.),
        ];

        let weights = vec![1.; 9];

        let id1 = ProcessUniqueId::new();
        let id2 = ProcessUniqueId::new();
        let id3 = ProcessUniqueId::new();

        let ids = vec![id1, id2, id3, id1, id2, id3, id1, id2, id3];

        let partition = Partition::from_ids(&points, &weights, ids);

        for part in partition.parts() {
            println!("new part of weight {}", part.total_weight());
            for (p, _) in part.iter() {
                println!("{}", p);
            }
        }

        panic!()
    }
}
