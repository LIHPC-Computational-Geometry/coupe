use super::Algorithm;
use super::Problem;
use anyhow::Result;
use mesh_io::weight;
use metis::Idx;

pub struct Recursive {
    pub part_count: Idx,
}

impl<const D: usize> Algorithm<D> for Recursive {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        let weights = match &problem.weights {
            weight::Array::Integers(is) => is,
            weight::Array::Floats(_) => anyhow::bail!("MeTiS does not support float weights"),
        };
        if weights.is_empty() {
            return Ok(());
        }
        let ncon = weights[0].len() as Idx;
        let mut weights: Vec<_> = weights.iter().flatten().map(|i| *i as Idx).collect();

        let (xadj, adjncy, _) = problem.adjacency.view().into_raw_storage();
        let mut xadj: Vec<_> = xadj.iter().map(|i| *i as Idx).collect();
        let mut adjncy: Vec<_> = adjncy.iter().map(|i| *i as Idx).collect();

        let mut metis_partition = vec![0; partition.len()];
        metis::Graph::new(ncon, self.part_count, &mut xadj, &mut adjncy)
            .set_vwgt(&mut weights)
            .part_recursive(&mut metis_partition)?;
        for (dst, src) in partition.iter_mut().zip(&mut metis_partition) {
            *dst = *src as usize;
        }
        Ok(())
    }
}

pub struct KWay {
    pub part_count: Idx,
}

impl<const D: usize> Algorithm<D> for KWay {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        let weights = match &problem.weights {
            weight::Array::Integers(is) => is,
            weight::Array::Floats(_) => anyhow::bail!("MeTiS does not support float weights"),
        };
        if weights.is_empty() {
            return Ok(());
        }
        let ncon = weights[0].len() as Idx;
        let mut weights: Vec<_> = weights.iter().flatten().map(|i| *i as Idx).collect();

        let (xadj, adjncy, _) = problem.adjacency.view().into_raw_storage();
        let mut xadj: Vec<_> = xadj.iter().map(|i| *i as Idx).collect();
        let mut adjncy: Vec<_> = adjncy.iter().map(|i| *i as Idx).collect();

        let mut metis_partition = vec![0; partition.len()];
        metis::Graph::new(ncon, self.part_count, &mut xadj, &mut adjncy)
            .set_vwgt(&mut weights)
            .part_kway(&mut metis_partition)?;
        for (dst, src) in partition.iter_mut().zip(&mut metis_partition) {
            *dst = *src as usize;
        }
        Ok(())
    }
}
