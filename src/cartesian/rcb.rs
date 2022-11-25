use super::Grid;
use super::SubGrid;
use num_traits::AsPrimitive;
use num_traits::Num;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::iter::Sum;

#[derive(Debug)]
pub enum IterationResult {
    Whole,
    Split {
        position: usize,
        left: Box<Self>,
        right: Box<Self>,
    },
}

impl IterationResult {
    pub fn part_of<const D: usize>(&self, pos: [usize; D], mut start_coord: usize) -> usize {
        let mut it = self;
        let mut part_id = 0;
        while let Self::Split {
            position,
            left,
            right,
        } = it
        {
            if pos[start_coord] < *position {
                part_id *= 2;
                it = left;
            } else {
                part_id = 2 * part_id + 1;
                it = right;
            }
            start_coord = (start_coord + 1) % D;
        }
        part_id
    }
}

const TOLERANCE: f64 = 0.01;

#[derive(Debug)]
struct WeightedMedian<W> {
    position: usize,
    left_weight: W,
}

fn weighted_median<W>(weights: &[W], total_weight: W) -> WeightedMedian<W>
where
    W: Send + Sync + PartialOrd + Num + Sum + AsPrimitive<f64>,
    f64: AsPrimitive<W>,
{
    let ideal_part_weight: f64 = total_weight.as_() / 2.0;
    let min_part_weight: W = (ideal_part_weight * (1.0 - TOLERANCE)).as_();
    let max_part_weight: W = (ideal_part_weight * (1.0 + TOLERANCE)).as_();
    let mut min = 0;
    let mut max = weights.len();
    let mut left_weight = W::zero();
    loop {
        let chunk_size = usize::max(1, (max - min) / rayon::current_num_threads());
        let chunk_weights: Vec<W> = weights[min..max]
            .par_iter()
            .fold_chunks(chunk_size, W::zero, |sum, w| sum + *w)
            .collect();
        let prefix_chunk_weights = chunk_weights.into_iter().enumerate().scan(
            W::zero(),
            move |prefix_sum, (chunk_idx, chunk_weight)| {
                *prefix_sum = *prefix_sum + chunk_weight;
                let chunk_start = min + chunk_idx * chunk_size;
                let prefix_chunk_weight = left_weight + *prefix_sum;
                Some((chunk_start, prefix_chunk_weight))
            },
        );
        for (position, prefix_chunk_weight) in prefix_chunk_weights {
            if prefix_chunk_weight < min_part_weight {
                min = position;
                left_weight = prefix_chunk_weight;
            } else if max_part_weight < prefix_chunk_weight {
                max = position;
                break;
            } else {
                return WeightedMedian {
                    position,
                    left_weight: prefix_chunk_weight,
                };
            }
        }
        if min + 1 >= max {
            return WeightedMedian {
                position: min,
                left_weight,
            };
        }
    }
}

pub(super) fn recurse_2d<W>(
    grid: Grid<2>,
    subgrid: SubGrid<2>,
    weights: &[W],
    total_weight: W,
    iter_count: usize,
    coord: usize,
) -> IterationResult
where
    W: Send + Sync + PartialOrd + Num + Sum + AsPrimitive<f64>,
    f64: AsPrimitive<W>,
{
    if subgrid.size[coord] == 0 || iter_count == 0 {
        return IterationResult::Whole;
    }

    let axis_weights: Vec<W> = if coord == 0 {
        subgrid
            .axis(0)
            .into_par_iter()
            .map(|x| {
                let s: W = subgrid
                    .axis(1)
                    .map(|y| weights[grid.index_of([x, y])])
                    .sum();
                s
            })
            .collect()
    } else {
        subgrid
            .axis(1)
            .into_par_iter()
            .map(|y| {
                let s: W = subgrid
                    .axis(0)
                    .map(|x| weights[grid.index_of([x, y])])
                    .sum();
                s
            })
            .collect()
    };

    let split = weighted_median(&axis_weights, total_weight);

    let split_position = split.position + subgrid.offset[coord];
    let left_weight = split.left_weight;
    let right_weight = total_weight - left_weight;

    let (left_grid, right_grid) = subgrid.split_at(coord, split_position);
    let (left, right) = rayon::join(
        || {
            recurse_2d(
                grid,
                left_grid,
                weights,
                left_weight,
                iter_count - 1,
                (coord + 1) % 2,
            )
        },
        || {
            recurse_2d(
                grid,
                right_grid,
                weights,
                right_weight,
                iter_count - 1,
                (coord + 1) % 2,
            )
        },
    );

    IterationResult::Split {
        position: split_position,
        left: Box::new(left),
        right: Box::new(right),
    }
}

pub(super) fn recurse_3d<W>(
    grid: Grid<3>,
    subgrid: SubGrid<3>,
    weights: &[W],
    total_weight: W,
    iter_count: usize,
    coord: usize,
) -> IterationResult
where
    W: Send + Sync + PartialOrd + Num + Sum + AsPrimitive<f64>,
    f64: AsPrimitive<W>,
{
    if subgrid.size[coord] == 0 || iter_count == 0 {
        return IterationResult::Whole;
    }

    let axis_weights: Vec<W> = if coord == 0 {
        subgrid
            .axis(0)
            .into_par_iter()
            .map(|x| {
                let s: W = subgrid
                    .axis(1)
                    .flat_map(|y| {
                        subgrid
                            .axis(2)
                            .map(move |z| weights[grid.index_of([x, y, z])])
                    })
                    .sum();
                s
            })
            .collect()
    } else if coord == 1 {
        subgrid
            .axis(1)
            .into_par_iter()
            .map(|y| {
                let s: W = subgrid
                    .axis(2)
                    .flat_map(|z| {
                        subgrid
                            .axis(0)
                            .map(move |x| weights[grid.index_of([x, y, z])])
                    })
                    .sum();
                s
            })
            .collect()
    } else {
        subgrid
            .axis(2)
            .into_par_iter()
            .map(|z| {
                let s: W = subgrid
                    .axis(0)
                    .flat_map(|x| {
                        subgrid
                            .axis(1)
                            .map(move |y| weights[grid.index_of([x, y, z])])
                    })
                    .sum();
                s
            })
            .collect()
    };

    let split = weighted_median(&axis_weights, total_weight);

    let split_position = split.position + subgrid.offset[coord];
    let left_weight = split.left_weight;
    let right_weight = total_weight - left_weight;

    let (left_grid, right_grid) = subgrid.split_at(coord, split_position);
    let (left, right) = rayon::join(
        || {
            recurse_3d(
                grid,
                left_grid,
                weights,
                left_weight,
                iter_count - 1,
                (coord + 1) % 3,
            )
        },
        || {
            recurse_3d(
                grid,
                right_grid,
                weights,
                right_weight,
                iter_count - 1,
                (coord + 1) % 3,
            )
        },
    );

    IterationResult::Split {
        position: split_position,
        left: Box::new(left),
        right: Box::new(right),
    }
}
