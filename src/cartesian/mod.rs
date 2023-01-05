use crate::topology::Topology;
use num_traits::AsPrimitive;
use num_traits::Num;
use num_traits::One;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashMap;
use std::fmt;
use std::iter::Sum;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::Range;

mod rcb;

/// Representation of a cartesian mesh.
///
/// Coupe can partition grids (also called cartesian meshes) faster and
/// consuming less memory than unstructured meshes.
///
/// # Example
///
/// You can feed grids to topologic algorithms, thanks to the [`Topology`]
/// trait. You can also partition them directly with, eg., RCB:
///
/// ```
/// # use coupe::Grid;
/// // Define a 2-by-2 grid.
/// let side = std::num::NonZeroUsize::new(2).unwrap();
/// let grid = Grid::new_2d(side, side);
///
/// // All cells have the same weight.
/// let mut partition = [0; 4];
/// let weights = [1.0; 4];
///
/// // Run 2 iterations of RCB.
/// grid.rcb(&mut partition, &weights, 2);
///
/// // There are 4 parts, their order is unspecified.
/// partition.sort();
/// assert_eq!(partition, [0, 1, 2, 3]);
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Grid<const D: usize> {
    size: [NonZeroUsize; D],
}

impl<const D: usize> Grid<D> {
    /// The subgrid that spans over all of the given grid.
    fn into_subgrid(self) -> SubGrid<D> {
        SubGrid {
            size: self.size.map(usize::from),
            offset: [0; D],
        }
    }

    /// The number of cells in the grid.
    fn len(&self) -> usize {
        self.size.iter().cloned().map(usize::from).product()
    }

    /// The spacial position of a cell, given its memory index.
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

    /// The memory index of a cell, given its spacial position.
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
    /// Define a new 2D grid.
    pub fn new_2d(width: NonZeroUsize, height: NonZeroUsize) -> Self {
        Self {
            size: [width, height],
        }
    }

    /// Run RCB on the 2D grid.
    ///
    /// Weights and partition indices are row major.
    pub fn rcb<W>(self, partition: &mut [usize], weights: &[W], iter_count: usize)
    where
        W: Send + Sync + PartialOrd + Num + Sum + AsPrimitive<f64>,
        f64: AsPrimitive<W>,
        W: num_traits::NumAssign,
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
            *p = iters.part_of(pos, 1);
        });

        let part_count = usize::pow(2, iter_count as u32);
        let mut part_loads = crate::imbalance::compute_parts_load(
            partition,
            part_count,
            weights.par_iter().cloned(),
        );
        let compute_imbalance = |part_loads: &[W]| {
            part_loads
                .iter()
                .map(|pl| {
                    let ideal_part_weight = total_weight / (part_count as f64).as_();
                    (*pl - ideal_part_weight) / ideal_part_weight
                })
                .max_by(crate::partial_cmp)
                .unwrap()
        };
        let check_move_imb = |part_loads: &mut [W], src: usize, dst: usize, weight: W| {
            part_loads[src] -= weight;
            part_loads[dst] += weight;
            let new_imbalance = compute_imbalance(&part_loads);
            part_loads[src] += weight;
            part_loads[dst] -= weight;
            new_imbalance
        };
        let imbalance = compute_imbalance(&part_loads);

        println!("{}", iters.fmt_svg(self, 1));
        let segs = iters.segments_2d(self, 1);

        // Testing horizontal segments for imbalance.
        for seg in &segs.c[0] {
            if seg.at != 0 {
                // Test if we can move segment down.
                let moved_cells = (seg.start..seg.end).map(|x| [x, seg.at - 1]);

                let src_part = partition[self.index_of([seg.start, seg.at - 1])];
                let dst_part = partition[self.index_of([seg.start, seg.at])];
                debug_assert_ne!(src_part, dst_part);

                let moved_weight = moved_cells.map(|pos| weights[self.index_of(pos)]).sum();
                let new_imbalance =
                    check_move_imb(&mut part_loads, src_part, dst_part, moved_weight);
                if new_imbalance < imbalance {
                    eprintln!("Found imb horiz move: {seg:?}");
                }
            }
            if seg.at + 1 >= usize::from(self.size[1]) {
                // Test if we can move segment up.
                let moved_cells = (seg.start..seg.end).map(|x| [x, seg.at]);

                let src_part = partition[self.index_of([seg.start, seg.at])];
                let dst_part = partition[self.index_of([seg.start, seg.at - 1])];
                debug_assert_ne!(src_part, dst_part);

                let moved_weight = moved_cells.map(|pos| weights[self.index_of(pos)]).sum();
                let new_imbalance =
                    check_move_imb(&mut part_loads, src_part, dst_part, moved_weight);
                if new_imbalance < imbalance {
                    eprintln!("Found imb horiz move: {seg:?}");
                }
            }
        }

        // Testing vertical segments for imbalance.
        for seg in &segs.c[1] {
            if seg.at != 0 {
                // Test if we can move segment left.
                let moved_cells = (seg.start..seg.end).map(|y| [seg.at - 1, y]);

                let src_part = partition[self.index_of([seg.at - 1, seg.start])];
                let dst_part = partition[self.index_of([seg.at, seg.start])];
                debug_assert_ne!(src_part, dst_part);

                let moved_weight = moved_cells.map(|pos| weights[self.index_of(pos)]).sum();
                let new_imbalance =
                    check_move_imb(&mut part_loads, src_part, dst_part, moved_weight);
                if new_imbalance < imbalance {
                    eprintln!("Found imb verti move: {seg:?}");
                }
            }
            if seg.at + 1 >= usize::from(self.size[0]) {
                // Test if we can move segment right.
                let moved_cells = (seg.start..seg.end).map(|y| [seg.at, y]);

                let src_part = partition[self.index_of([seg.at, seg.start])];
                let dst_part = partition[self.index_of([seg.at - 1, seg.start])];
                debug_assert_ne!(src_part, dst_part);

                let moved_weight = moved_cells.map(|pos| weights[self.index_of(pos)]).sum();
                let new_imbalance =
                    check_move_imb(&mut part_loads, src_part, dst_part, moved_weight);
                if new_imbalance < imbalance {
                    eprintln!("Found imb verti move: {seg:?}");
                }
            }
        }

        // Testing horizontal segments for lambda cut.
        for seg in &segs.c[0] {
            if seg.at >= 2 {
                // Test if we can move segment down.
                let a = (seg.start..seg.end).map(|x| [x, seg.at]);
                let b = (seg.start..seg.end).map(|x| [x, seg.at - 2]);

                let a_weight: W = a.map(|pos| weights[self.index_of(pos)]).sum();
                let b_weight: W = b.map(|pos| weights[self.index_of(pos)]).sum();

                if b_weight < a_weight {
                    eprintln!("Found lambda horiz move: {seg:?}");
                }
            }
            if seg.at + 2 >= usize::from(self.size[1]) {
                // Test if we can move segment up.
                let a = (seg.start..seg.end).map(|x| [x, seg.at - 1]);
                let b = (seg.start..seg.end).map(|x| [x, seg.at + 1]);

                let a_weight: W = a.map(|pos| weights[self.index_of(pos)]).sum();
                let b_weight: W = b.map(|pos| weights[self.index_of(pos)]).sum();

                if b_weight < a_weight {
                    eprintln!("Found lambda horiz move: {seg:?}");
                }
            }
        }

        // Testing vertical segments for lambda cut.
        for seg in &segs.c[1] {
            if seg.at >= 2 {
                // Test if we can move segment down.
                let a = (seg.start..seg.end).map(|y| [seg.at, y]);
                let b = (seg.start..seg.end).map(|y| [seg.at - 2, y]);

                let a_weight: W = a.map(|pos| weights[self.index_of(pos)]).sum();
                let b_weight: W = b.map(|pos| weights[self.index_of(pos)]).sum();

                if b_weight < a_weight {
                    eprintln!("Found lambda verti move: {seg:?}");
                }
            }
            if seg.at + 2 >= usize::from(self.size[0]) {
                // Test if we can move segment up.
                let a = (seg.start..seg.end).map(|y| [seg.at - 1, y]);
                let b = (seg.start..seg.end).map(|y| [seg.at + 1, y]);

                let a_weight: W = a.map(|pos| weights[self.index_of(pos)]).sum();
                let b_weight: W = b.map(|pos| weights[self.index_of(pos)]).sum();

                if b_weight < a_weight {
                    eprintln!("Found lambda verti move: {seg:?}");
                }
            }
        }
    }
}

impl Grid<3> {
    /// Define a new 3D grid.
    pub fn new_3d(width: NonZeroUsize, height: NonZeroUsize, depth: NonZeroUsize) -> Self {
        Self {
            size: [width, height, depth],
        }
    }

    /// Run RCB on the 3D grid.
    ///
    /// Weights and partition indices are row-then-column major.
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
            *p = iters.part_of(pos, 1);
        });
    }
}

/// A specific, rectangular region of a grid.
#[derive(Copy, Clone, Debug)]
struct SubGrid<const D: usize> {
    size: [usize; D],
    offset: [usize; D],
}

impl<const D: usize> SubGrid<D> {
    /// The set of indices for the given axis.
    ///
    /// # Panics
    ///
    /// This function panics if `coord` is `D` or larger.
    fn axis(&self, coord: usize) -> Range<usize> {
        let size = self.size[coord];
        let offset = self.offset[coord];
        offset..offset + size
    }

    /// Split the subgrid into two along an axis.
    ///
    /// # Panics
    ///
    /// This function panics in the following cases:
    ///
    /// - `coord` is `D` or larger, or
    /// - `at` is not in `self.axis(coord)`.
    fn split_at(self, coord: usize, at: usize) -> (SubGrid<D>, SubGrid<D>) {
        let mut low = self;
        let mut high = self;
        low.size[coord] = at - self.offset[coord];
        high.size[coord] -= at - self.offset[coord];
        high.offset[coord] = at;
        (low, high)
    }
}

/// An iterator over the neighbors of a grid cell.
///
/// This type is returned by [`Grid::neighbors`].
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

fn transpose<T>(p: [T; 2]) -> [T; 2] {
    let [a, b] = p;
    [b, a]
}

#[derive(Debug)]
pub enum SplitTree {
    Whole,
    Split {
        position: usize,
        left: Box<Self>,
        right: Box<Self>,
    },
}

impl SplitTree {
    fn part_of<const D: usize>(&self, pos: [usize; D], mut start_coord: usize) -> usize {
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

    pub fn fmt_svg(&self, grid: Grid<2>, start_coord: usize) -> impl fmt::Display + '_ {
        struct ShowSvg<'a> {
            tree: &'a SplitTree,
            grid: Grid<2>,
            start_coord: usize,
        }

        fn print_splits(
            f: &mut fmt::Formatter<'_>,
            sg: SubGrid<2>,
            tree: &SplitTree,
            coord: usize,
            iter: usize,
        ) -> fmt::Result {
            let SplitTree::Split { position, left, right } = tree
            else { return Ok(()) };

            // Recurse before so that lines from first iterations are shown
            // above lines from the next ones.
            let (sg_left, sg_right) = sg.split_at(coord, *position);
            print_splits(f, sg_left, left, (coord + 1) % 2, iter + 1)?;
            print_splits(f, sg_right, right, (coord + 1) % 2, iter + 1)?;

            let Range { start, end } = sg.axis(1 - coord);
            let mut p1 = [*position, start];
            let mut p2 = [*position, end];
            if coord == 1 {
                p1 = transpose(p1);
                p2 = transpose(p2);
            }
            let color = match iter % 10 {
                0 => "maroon",
                1 => "green",
                2 => "red",
                3 => "lime",
                4 => "purple",
                5 => "olive",
                6 => "fuchsia",
                7 => "yellow",
                8 => "navy",
                _ => "blue",
            };
            writeln!(
                f,
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}"/>"#,
                p1[0], p1[1], p2[0], p2[1], color,
            )
        }

        impl fmt::Display for ShowSvg<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let [gwidth, gheight] = self.grid.size;
                let sg = self.grid.into_subgrid();

                writeln!(
                    f,
                    r#"<svg viewBox="0 0 {gwidth} {gheight}" xmlns="http://www.w3.org/2000/svg">"#
                )?;

                print_splits(f, sg, self.tree, self.start_coord, 0)?;

                for [x, y] in self.tree.joints_2d(self.grid, self.start_coord) {
                    writeln!(f, r#"<circle cx="{x}" cy="{y}" r="2" fill="black"/>"#)?;
                }

                writeln!(f, "</svg>")
            }
        }

        ShowSvg {
            tree: self,
            grid,
            start_coord,
        }
    }

    pub fn joints_2d(&self, grid: Grid<2>, start_coord: usize) -> impl Iterator<Item = [usize; 2]> {
        fn aux(
            joints: &mut HashMap<[usize; 2], usize>,
            tree: &SplitTree,
            sg: SubGrid<2>,
            coord: usize,
        ) {
            let SplitTree::Split { position, left, right } = tree
            else { return };

            let Range { start, end } = sg.axis(1 - coord);
            let mut p1 = [*position, start];
            let mut p2 = [*position, end];
            if coord == 1 {
                p1 = transpose(p1);
                p2 = transpose(p2);
            }

            *joints.entry(p1).or_default() += 1;
            *joints.entry(p2).or_default() += 1;

            let (sg_left, sg_right) = sg.split_at(coord, *position);
            aux(joints, left, sg_left, (coord + 1) % 2);
            aux(joints, right, sg_right, (coord + 1) % 2);
        }

        let mut joints = HashMap::new();

        aux(&mut joints, self, grid.into_subgrid(), start_coord);
        joints.retain(|_, occ| *occ > 1);

        joints.into_keys()
    }

    pub fn segments_2d(&self, grid: Grid<2>, start_coord: usize) -> Segments<2> {
        fn aux(c: &mut [Vec<Segment>; 2], tree: &SplitTree, sg: SubGrid<2>, coord: usize) {
            let SplitTree::Split { position, left, right } = tree
            else { return };

            let Range { start, end } = sg.axis(1 - coord);
            c[1 - coord].push(Segment {
                start,
                end,
                at: *position,
            });

            let (sg_left, sg_right) = sg.split_at(coord, *position);
            aux(c, left, sg_left, (coord + 1) % 2);
            aux(c, right, sg_right, (coord + 1) % 2);
        }

        let mut c = [Vec::new(), Vec::new()];
        aux(&mut c, self, grid.into_subgrid(), start_coord);

        for [x, y] in self.joints_2d(grid, start_coord) {
            for i in 0..c[0].len() {
                let seg = &c[0][i];
                if y != seg.at {
                    continue;
                }
                let Some((left, right)) = seg.split_at(x)
                else { continue };
                c[0][i] = left;
                c[0].push(right);
            }
            for i in 0..c[1].len() {
                let seg = &c[1][i];
                if x != seg.at {
                    continue;
                }
                let Some((left, right)) = seg.split_at(y)
                else { continue };
                c[1][i] = left;
                c[1].push(right);
            }
        }

        Segments { c, grid }
    }
}

#[derive(Debug)]
pub struct Segment {
    start: usize,
    end: usize,
    at: usize,
}

impl Segment {
    pub fn split_at(&self, pos: usize) -> Option<(Segment, Segment)> {
        if pos <= self.start || self.end <= pos {
            return None;
        }
        Some((
            Segment {
                start: self.start,
                end: pos,
                at: self.at,
            },
            Segment {
                start: pos,
                end: self.end,
                at: self.at,
            },
        ))
    }
}

#[derive(Debug)]
pub struct Segments<const D: usize> {
    c: [Vec<Segment>; D],
    grid: Grid<D>,
}

//impl Segments<2> {
//    pub fn moves(&self) -> impl Iterator<Item = Segment> {
//        // TODO
//    }
//}

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
