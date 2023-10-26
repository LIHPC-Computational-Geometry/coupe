use super::runner_error;
use super::Problem;
use super::ToRunner;
use anyhow::Context as _;
use mesh_io::weight;
use scotch::graph::Data;
use scotch::Graph;
use scotch::Num;

pub struct Standard {
    pub part_count: Num,
    pub strategy: scotch::Strategy,
}

impl<const D: usize> ToRunner<D> for Standard {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> super::Runner<'a> {
        let weights = match &problem.weights {
            weight::Array::Integers(is) => {
                if is.first().map_or(1, Vec::len) != 1 {
                    return runner_error("SCOTCH cannot do multi-criteria partitioning");
                }
                crate::zoom_in(is.iter().map(|v| Some(v[0])))
            }
            weight::Array::Floats(fs) => {
                if fs.first().map_or(1, Vec::len) != 1 {
                    return runner_error("SCOTCH cannot do multi-criteria partitioning");
                }
                crate::zoom_in(fs.iter().map(|v| Some(v[0])))
            }
        };

        let (xadj, adjncy, adjwgt) = problem.adjacency().into_raw_storage();
        let xadj: Vec<_> = xadj.iter().map(|i| *i as Num).collect();
        let adjncy: Vec<_> = adjncy.iter().map(|i| *i as Num).collect();
        let adjwgt = crate::zoom_in(adjwgt.iter().map(|v| Some(*v)));

        let arch = scotch::Architecture::complete(self.part_count as Num);

        let mut scotch_partition = vec![0; weights.len()];
        Box::new(move |partition| {
            let graph_data = Data::new(0, &xadj, &[], &weights, &[], &adjncy, &adjwgt);
            let mut graph = Graph::build(&graph_data).context("failed to build SCOTCH graph")?;
            graph.check().context("failed to build SCOTCH graph")?;
            graph
                .mapping(&arch, &mut scotch_partition)
                .compute(&mut self.strategy)
                .context("SCOTCH partitioning failed")?;
            for (dst, src) in partition.iter_mut().zip(&scotch_partition) {
                *dst = *src as usize;
            }
            Ok(None)
        })
    }
}

pub struct Remap {
    pub part_count: Num,
    pub strategy: scotch::Strategy,
}

impl<const D: usize> ToRunner<D> for Remap {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> super::Runner<'a> {
        let weights = match &problem.weights {
            weight::Array::Integers(is) => {
                if is.first().map_or(1, Vec::len) != 1 {
                    return runner_error("SCOTCH cannot do multi-criteria partitioning");
                }
                crate::zoom_in(is.iter().map(|v| Some(v[0])))
            }
            weight::Array::Floats(fs) => {
                if fs.first().map_or(1, Vec::len) != 1 {
                    return runner_error("SCOTCH cannot do multi-criteria partitioning");
                }
                crate::zoom_in(fs.iter().map(|v| Some(v[0])))
            }
        };

        let (xadj, adjncy, adjwgt) = problem.adjacency().into_raw_storage();
        let xadj: Vec<_> = xadj.iter().map(|i| *i as Num).collect();
        let adjncy: Vec<_> = adjncy.iter().map(|i| *i as Num).collect();
        let adjwgt = crate::zoom_in(adjwgt.iter().map(|v| Some(*v)));

        let arch = scotch::Architecture::complete(self.part_count as Num);

        let mut out_partition = vec![0; weights.len()];
        Box::new(move |partition| {
            let graph_data = Data::new(0, &xadj, &[], &weights, &[], &adjncy, &adjwgt);
            let mut graph = Graph::build(&graph_data).context("failed to build SCOTCH graph")?;
            graph.check().context("failed to build SCOTCH graph")?;
            let mut in_partition: Vec<_> = partition
                .iter()
                .map(|p| scotch::Num::try_from(*p).unwrap())
                .collect();
            graph
                .remap(
                    &arch,
                    &mut in_partition,
                    &mut self.strategy,
                    &mut out_partition,
                )
                .context("SCOTCH remaping failed")?;
            for (dst, src) in partition.iter_mut().zip(&in_partition) {
                *dst = *src as usize;
            }
            Ok(None)
        })
    }
}
