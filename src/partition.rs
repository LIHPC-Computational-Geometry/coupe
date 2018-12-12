//! Utilities to manipulate partitions

use itertools::Itertools;
use rayon::prelude::*;
use snowflake::ProcessUniqueId;
use std::sync::atomic::{self, AtomicPtr};

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
        W: std::ops::Sub<Output = W> + std::iter::Sum + Clone,
    {
        let total_weights = self
            .parts()
            .map(|part| part.total_weight())
            .collect::<Vec<_>>();
        total_weights
            .iter()
            .flat_map(|w1| total_weights.iter().map(move |w2| (*w1 - *w2).abs()))
            .max_by(|a, b| a.partial_cmp(&b).unwrap())
            // if the partition is empty, then there is the imbalance is null
            .unwrap_or(0.)
    }
}

pub struct Part<'a, P, W> {
    partition: &'a Partition<'a, P, W>,
    indices: Vec<usize>,
}

impl<'a, P, W> Part<'a, P, W> {
    pub fn total_weight(&self) -> W
    where
        W: std::iter::Sum + Clone,
    {
        self.indices
            .iter()
            .map(|idx| &self.partition.weights[*idx])
            .cloned()
            .sum()
    }
}

impl<'a, P, W> Iterator for Part<'a, P, W> {
    type Item = (&'a P, &'a W);
    fn next(&mut self) -> Option<Self::Item> {
        self.indices
            .pop()
            .and_then(|idx| Some((&self.partition.points[idx], &self.partition.weights[idx])))
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
            for (p, _) in part {
                println!("{}", p);
            }
        }

        panic!()
    }
}
