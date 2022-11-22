use super::Problem;
use super::Runner;
use super::ToRunner;
use anyhow::Context;
use mesh_io::weight;
use metis::Idx;

pub struct Recursive {
    pub part_count: Idx,
    pub tolerance: Option<f64>,
}

impl<const D: usize> ToRunner<D> for Recursive {
    fn to_runner<'a>(&'a mut self, problem: &Problem<D>) -> Runner<'a> {
        let weights = problem.weights();
        let (ncon, mut weights) = match &*weights {
            weight::Array::Integers(is) => {
                let ncon = is.first().map_or(1, Vec::len) as Idx;
                let weights = crate::zoom_in(is.iter().map(|v| v.iter().cloned()));
                (ncon, weights)
            }
            weight::Array::Floats(fs) => {
                let ncon = fs.first().map_or(1, Vec::len) as Idx;
                let weights = crate::zoom_in(fs.iter().map(|v| v.iter().cloned()));
                (ncon, weights)
            }
        };

        let adjacency = problem.adjacency();
        let (xadj, adjncy, adjwgt) = adjacency.view().into_raw_storage();
        let mut xadj: Vec<_> = xadj.iter().map(|i| *i as Idx).collect();
        let mut adjncy: Vec<_> = adjncy.iter().map(|i| *i as Idx).collect();
        let mut adjwgt: Vec<_> = crate::zoom_in(adjwgt.iter().map(|v| Some(*v)));

        let tolerance = self.tolerance.map(|f| (f * 1000.0) as Idx);

        let mut metis_partition = vec![0; weights.len()];
        Box::new(move |partition| {
            let mut graph = metis::Graph::new(ncon, self.part_count, &mut xadj, &mut adjncy)
                .set_vwgt(&mut weights)
                .set_adjwgt(&mut adjwgt);
            if let Some(tolerance) = tolerance {
                graph = graph.set_option(metis::option::UFactor(tolerance));
            }
            graph
                .part_recursive(&mut metis_partition)
                .context("METIS partitioning failed")?;
            for (dst, src) in partition.iter_mut().zip(&metis_partition) {
                *dst = *src as usize;
            }
            Ok(None)
        })
    }
}

pub struct KWay {
    pub part_count: Idx,
    pub tolerance: Option<f64>,
}

impl<const D: usize> ToRunner<D> for KWay {
    fn to_runner<'a>(&'a mut self, problem: &Problem<D>) -> Runner<'a> {
        let weights = problem.weights();
        let (ncon, mut weights) = match &*weights {
            weight::Array::Integers(is) => {
                let ncon = is.first().map_or(1, Vec::len) as Idx;
                let weights = crate::zoom_in(is.iter().map(|v| v.iter().cloned()));
                (ncon, weights)
            }
            weight::Array::Floats(fs) => {
                let ncon = fs.first().map_or(1, Vec::len) as Idx;
                let weights = crate::zoom_in(fs.iter().map(|v| v.iter().cloned()));
                (ncon, weights)
            }
        };

        let adjacency = problem.adjacency();
        let (xadj, adjncy, adjwgt) = adjacency.view().into_raw_storage();
        let mut xadj: Vec<_> = xadj.iter().map(|i| *i as Idx).collect();
        let mut adjncy: Vec<_> = adjncy.iter().map(|i| *i as Idx).collect();
        let mut adjwgt: Vec<_> = crate::zoom_in(adjwgt.iter().map(|v| Some(*v)));

        let tolerance = self.tolerance.map(|f| (f * 1000.0) as Idx);

        let mut metis_partition = vec![0; weights.len()];
        Box::new(move |partition| {
            let mut graph = metis::Graph::new(ncon, self.part_count, &mut xadj, &mut adjncy)
                .set_vwgt(&mut weights)
                .set_adjwgt(&mut adjwgt);
            if let Some(tolerance) = tolerance {
                graph = graph.set_option(metis::option::UFactor(tolerance));
            }
            graph
                .part_kway(&mut metis_partition)
                .context("METIS partitioning failed")?;
            for (dst, src) in partition.iter_mut().zip(&metis_partition) {
                *dst = *src as usize;
            }
            Ok(None)
        })
    }
}
