use std::iter::Sum;
use std::num::NonZeroUsize;
use std::ops::Range;

use num_traits::AsPrimitive;
use num_traits::Num;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

mod rcb;

#[derive(Copy, Clone, Debug)]
pub struct Grid<const D: usize> {
    size: [NonZeroUsize; D],
}

impl<const D: usize> Grid<D> {
    fn into_subgrid(self) -> SubGrid<D> {
        SubGrid {
            size: self.size.map(usize::from),
            offset: [0; D],
        }
    }

    fn position_of(&self, mut i: usize) -> [usize; D] {
        let mut pos = [0; D];
        match D {
            2 => {
                let width = self.size[0];
                pos[0] = i % width;
                pos[1] = i / width;
            }
            3 => {
                let width = self.size[0];
                let height = self.size[1];
                pos[0] = i % width;
                pos[1] = (i / width) % height;
                pos[2] = i / width / height;
            }
            _ => {
                for (s, p) in self.size.into_iter().zip(&mut pos) {
                    *p = i % s;
                    i = i / s;
                }
            }
        }
        pos
    }

    fn index_of(&self, pos: [usize; D]) -> usize {
        match D {
            2 => {
                let x = pos[0];
                let y = pos[1];
                let width = usize::from(self.size[0]);
                x + width * y
            }
            3 => {
                let x = pos[0];
                let y = pos[1];
                let z = pos[2];
                let width = usize::from(self.size[0]);
                let height = usize::from(self.size[1]);
                x + width * (y + height * z)
            }
            _ => self
                .size
                .into_iter()
                .zip(pos)
                .scan(1, |prefix, (s, p)| {
                    let a = *prefix * p;
                    *prefix *= usize::from(s);
                    Some(a)
                })
                .sum(),
        }
    }
}

impl Grid<2> {
    pub fn new_2d(width: NonZeroUsize, height: NonZeroUsize) -> Self {
        Self {
            size: [width, height],
        }
    }

    pub fn rcb<W>(self, partition: &mut [usize], weights: &[W], iter_count: usize)
    where
        W: Send + Sync + PartialOrd + Num + Sum + AsPrimitive<f64>,
        f64: AsPrimitive<W>,
    {
        let total_weight: W = weights.par_iter().cloned().sum();
        let iters = rcb::recurse_2d(
            self,
            self.into_subgrid(),
            weights,
            total_weight,
            iter_count,
            1,
        );
        partition.par_iter_mut().enumerate().for_each(|(i, p)| {
            let pos = self.position_of(i);
            *p = iters.part_of(pos, 0);
        });
    }
}

impl Grid<3> {
    pub fn new_3d(width: NonZeroUsize, height: NonZeroUsize, depth: NonZeroUsize) -> Self {
        Self {
            size: [width, height, depth],
        }
    }

    pub fn rcb<W>(self, partition: &mut [usize], weights: &[W], iter_count: usize)
    where
        W: Send + Sync + PartialOrd + Num + Sum + AsPrimitive<f64>,
        f64: AsPrimitive<W>,
    {
        let total_weight: W = weights.par_iter().cloned().sum();
        let iters = rcb::recurse_3d(
            self,
            self.into_subgrid(),
            weights,
            total_weight,
            iter_count,
            1,
        );
        partition.par_iter_mut().enumerate().for_each(|(i, p)| {
            let pos = self.position_of(i);
            *p = iters.part_of(pos, 0);
        });
    }
}

#[derive(Copy, Clone, Debug)]
struct SubGrid<const D: usize> {
    size: [usize; D],
    offset: [usize; D],
}

impl<const D: usize> SubGrid<D> {
    fn axis(&self, coord: usize) -> Range<usize> {
        let size = self.size[coord];
        let offset = self.offset[coord];
        offset..offset + size
    }

    fn split_at(self, coord: usize, at: usize) -> (SubGrid<D>, SubGrid<D>) {
        let mut low = self;
        let mut high = self;
        low.size[coord] = at - self.offset[coord];
        high.size[coord] -= at - self.offset[coord];
        high.offset[coord] = at;
        (low, high)
    }
}
