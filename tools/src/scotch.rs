use super::runner_error;
use super::Problem;
use super::ToRunner;
use anyhow::Context as _;
use mesh_io::weight;
use scotch::Num;

pub struct Standard {
    pub part_count: Num,
}

impl<const D: usize> ToRunner<D> for Standard {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> super::Runner<'a> {
        let weights = match &problem.weights {
            weight::Array::Integers(is) => is,
            weight::Array::Floats(_) => {
                return runner_error("SCOTCH does not support float weights");
            }
        };
        if weights.first().map_or(1, Vec::len) != 1 {
            return runner_error("SCOTCH cannot do multi-criteria partitioning");
        }
        let weights: Vec<_> = weights.iter().map(|i| i[0] as Num).collect();

        let (xadj, adjncy, _) = problem.adjacency.view().into_raw_storage();
        let xadj: Vec<_> = xadj.iter().map(|i| *i as Num).collect();
        let adjncy: Vec<_> = adjncy.iter().map(|i| *i as Num).collect();

        let mut strat = scotch::Strategy::new();
        let arch = scotch::Architecture::complete(self.part_count as Num);

        let mut scotch_partition = vec![0; weights.len()];
        Box::new(move |partition| {
            let graph_data = scotch::graph::Data::new(0, &xadj, &[], &weights, &[], &adjncy, &[]);
            let mut graph =
                scotch::Graph::build(&graph_data).context("failed to build SCOTCH graph")?;
            graph.check().context("failed to build SCOTCH graph")?;
            graph
                .mapping(&arch, &mut scotch_partition)
                .compute(&mut strat)
                .context("SCOTCH partitioning failed")?;
            for (dst, src) in partition.iter_mut().zip(&mut scotch_partition) {
                *dst = *src as usize;
            }
            Ok(())
        })
    }
}
