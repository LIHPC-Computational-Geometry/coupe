use crate::topology::Topology;
use num_traits::AsPrimitive;
use num_traits::Num;
use num_traits::One;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::iter::Sum;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::Range;

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

    fn len(&self) -> usize {
        self.size.iter().cloned().map(usize::from).product()
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

#[derive(Debug)]
pub struct GridNeighbors<const D: usize, E> {
    grid: Grid<D>,
    vertex: [usize; D],
    i: usize,
    _marker: PhantomData<E>,
}

impl<const D: usize, E> Iterator for GridNeighbors<D, E>
where
    E: One,
{
    type Item = (usize, E);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let i = self.i;
            if i >= 2 * D {
                return None;
            }
            self.i += 1;

            let mut neighbor = self.vertex;
            let axis = i / 2;
            let new_coord = if (i % 2) == 0 {
                usize::checked_sub(neighbor[axis], 1)
            } else {
                Some(neighbor[axis] + 1)
            };
            match new_coord {
                None => continue,
                Some(v) if v >= usize::from(self.grid.size[axis]) => continue,
                Some(v) => neighbor[axis] = v,
            }

            let neighbor_idx = self.grid.index_of(neighbor);
            return Some((neighbor_idx, E::one()));
        }
    }
}

impl<const D: usize, E> Topology<E> for Grid<D>
where
    E: One,
{
    type Neighbors<'a> = GridNeighbors<D, E> where Self: 'a;

    fn len(&self) -> usize {
        self.len()
    }

    fn neighbors(&self, vertex: usize) -> Self::Neighbors<'_> {
        GridNeighbors {
            grid: *self,
            vertex: self.position_of(vertex),
            i: 0,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_neighbors() {
        fn neighbors<const D: usize>(g: Grid<D>, vertex: usize) -> Vec<usize> {
            let mut ns: Vec<usize> = Topology::<usize>::neighbors(&g, vertex)
                .map(|(n, _)| n)
                .collect();
            ns.sort();
            ns
        }

        let side = NonZeroUsize::new(3).unwrap();

        // Grid ids:
        //
        //     0 -- 1 -- 2
        //     |    |    |
        //     3 -- 4 -- 5
        //     |    |    |
        //     6 -- 7 -- 8
        //
        let g = Grid::new_2d(side, side);
        assert_eq!(neighbors(g, 0), vec![1, 3]);
        assert_eq!(neighbors(g, 1), vec![0, 2, 4]);
        assert_eq!(neighbors(g, 2), vec![1, 5]);
        assert_eq!(neighbors(g, 3), vec![0, 4, 6]);
        assert_eq!(neighbors(g, 4), vec![1, 3, 5, 7]);
        assert_eq!(neighbors(g, 5), vec![2, 4, 8]);
        assert_eq!(neighbors(g, 6), vec![3, 7]);
        assert_eq!(neighbors(g, 7), vec![4, 6, 8]);
        assert_eq!(neighbors(g, 8), vec![5, 7]);

        // Grid ids:
        //
        //                                 I -- J -- K
        //                                 |    |    |
        //                   9 -- A -- B   L -- M -- N
        //                   |    |    |   |    |    |
        //     0 -- 1 -- 2   C -- D -- E   O -- P -- Q
        //     |    |    |   |    |    |
        //     3 -- 4 -- 5   F -- G -- H
        //     |    |    |
        //     6 -- 7 -- 8
        //
        let g = Grid::new_3d(side, side, side);
        assert_eq!(neighbors(g, 0), vec![1, 3, 9]);
        assert_eq!(neighbors(g, 1), vec![0, 2, 4, 10]);
        assert_eq!(neighbors(g, 2), vec![1, 5, 11]);
        assert_eq!(neighbors(g, 3), vec![0, 4, 6, 12]);
        assert_eq!(neighbors(g, 4), vec![1, 3, 5, 7, 13]);
        assert_eq!(neighbors(g, 5), vec![2, 4, 8, 14]);
        assert_eq!(neighbors(g, 6), vec![3, 7, 15]);
        assert_eq!(neighbors(g, 7), vec![4, 6, 8, 16]);
        assert_eq!(neighbors(g, 8), vec![5, 7, 17]);

        assert_eq!(neighbors(g, 9), vec![0, 10, 12, 18]);
        assert_eq!(neighbors(g, 10), vec![1, 9, 11, 13, 19]);
        assert_eq!(neighbors(g, 11), vec![2, 10, 14, 20]);
        assert_eq!(neighbors(g, 12), vec![3, 9, 13, 15, 21]);
        assert_eq!(neighbors(g, 13), vec![4, 10, 12, 14, 16, 22]);
        assert_eq!(neighbors(g, 14), vec![5, 11, 13, 17, 23]);
        assert_eq!(neighbors(g, 15), vec![6, 12, 16, 24]);
        assert_eq!(neighbors(g, 16), vec![7, 13, 15, 17, 25]);
        assert_eq!(neighbors(g, 17), vec![8, 14, 16, 26]);

        assert_eq!(neighbors(g, 18), vec![9, 19, 21]);
        assert_eq!(neighbors(g, 19), vec![10, 18, 20, 22]);
        assert_eq!(neighbors(g, 20), vec![11, 19, 23]);
        assert_eq!(neighbors(g, 21), vec![12, 18, 22, 24]);
        assert_eq!(neighbors(g, 22), vec![13, 19, 21, 23, 25]);
        assert_eq!(neighbors(g, 23), vec![14, 20, 22, 26]);
        assert_eq!(neighbors(g, 24), vec![15, 21, 25]);
        assert_eq!(neighbors(g, 25), vec![16, 22, 24, 26]);
        assert_eq!(neighbors(g, 26), vec![17, 23, 25]);
    }

    #[test]
    fn test_grid_edge_cut() {
        let side = NonZeroUsize::new(3).unwrap();
        let g = Grid::new_2d(side, side);
        let weights = [1; 9];

        // Grid ids and partition:
        //
        //     0 -- 1 -- 2    A -- A == B
        //     |    |    |    |    #    #
        //     3 -- 4 -- 5    A == B == A
        //     |    |    |    #    |    |
        //     6 -- 7 -- 8    B -- B == A
        //
        let partition = [0, 0, 1, 0, 1, 0, 1, 1, 0];
        assert_eq!(Topology::<usize>::edge_cut(&g, &partition), 7);
        assert_eq!(Topology::<usize>::lambda_cut(&g, &partition, weights), 8);

        // Grid ids and partition:
        //
        //     0 -- 1 -- 2    A -- A -- A
        //     |    |    |    #    #    #
        //     3 -- 4 -- 5    B == C -- C
        //     |    |    |    |    #    |
        //     6 -- 7 -- 8    B -- B == C
        //
        let partition = [0, 0, 0, 1, 2, 2, 1, 1, 2];
        assert_eq!(Topology::<usize>::edge_cut(&g, &partition), 6);
        assert_eq!(Topology::<usize>::lambda_cut(&g, &partition, weights), 10);
    }

    #[test]
    fn test_split_at() {
        let side = NonZeroUsize::new(6).unwrap();
        let grid = Grid::new_2d(side, side);
        let subgrid = grid.into_subgrid();
        let (low, high) = subgrid.split_at(0, 3);
        assert_eq!(low.size, [3, 6]);
        assert_eq!(high.size, [3, 6]);
        assert_eq!(low.offset, [0, 0]);
        assert_eq!(high.offset, [3, 0]);
    }
}
