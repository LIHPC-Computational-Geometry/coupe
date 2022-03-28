use super::Algorithm;
use super::Problem;
use anyhow::Context as _;
use anyhow::Result;
use mesh_io::weight;
use scotch::Num;

pub struct Standard {
    pub part_count: Num,
}

impl<const D: usize> Algorithm<D> for Standard {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        let weights = match &problem.weights {
            weight::Array::Integers(is) => is,
            weight::Array::Floats(_) => anyhow::bail!("SCOTCH does not support float weights"),
        };
        if weights.is_empty() {
            return Ok(());
        }
        if weights[0].len() != 1 {
            anyhow::bail!("SCOTCH cannot do multi-criteria partitioning");
        }
        let weights: Vec<_> = weights.iter().map(|i| i[0] as Num).collect();

        let (xadj, adjncy, _) = problem.adjacency.view().into_raw_storage();
        let xadj: Vec<_> = xadj.iter().map(|i| *i as Num).collect();
        let adjncy: Vec<_> = adjncy.iter().map(|i| *i as Num).collect();

        let mut strat = scotch::Strategy::new();
        let arch = scotch::Architecture::complete(self.part_count as Num);
        let graph_data = scotch::graph::Data::new(0, &xadj, &[], &weights, &[], &adjncy, &[]);
        let mut graph =
            scotch::Graph::build(&graph_data).context("failed to build SCOTCH graph")?;
        graph.check().context("failed to build SCOTCH graph")?;

        let mut scotch_partition = vec![0; partition.len()];
        graph
            .mapping(&arch, &mut scotch_partition)
            .compute(&mut strat)
            .context("SCOTCH partitioning failed")?;
        // TODO use mapping?
        for (dst, src) in partition.iter_mut().zip(&mut scotch_partition) {
            *dst = *src as usize;
        }
        Ok(())
    }
}
