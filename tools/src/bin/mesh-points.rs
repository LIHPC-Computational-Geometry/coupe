use anyhow::Result;
use mesh_io::ElementType;
use mesh_io::Mesh;
use std::io;

const USAGE: &str = "Usage: mesh-points [options] [in-mesh [out-plot]] <in.mesh >out.plot";

fn write_points<const D: usize>(mut w: impl io::Write, mesh: &Mesh, with_ids: bool) -> Result<()> {
    let points = coupe_tools::barycentres::<D>(mesh);
    let ids: Box<dyn Iterator<Item = isize>> = if with_ids {
        Box::new(
            match mesh
                .topology()
                .iter()
                .map(|(el_type, _, _)| el_type.dimension())
                .max()
            {
                Some(element_dim) => mesh
                    .elements()
                    .filter_map(|(element_type, _nodes, element_ref)| {
                        if element_type.dimension() != element_dim
                            || element_type == ElementType::Edge
                        {
                            return None;
                        }
                        Some(element_ref)
                    })
                    .collect(),
                None => Vec::new(),
            }
            .into_iter(),
        )
    } else {
        Box::new(std::iter::repeat(0))
    };

    for (id, point) in ids.zip(points) {
        for coord in point.into_iter() {
            write!(w, "{coord} ")?;
        }
        if with_ids {
            write!(w, "{id}")?;
        }
        writeln!(w)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("", "with-ids", "add a column with the cell id");

    let matches = coupe_tools::parse_args(options, USAGE, 2)?;

    let mesh = coupe_tools::read_mesh(matches.free.get(0))?;
    let output = coupe_tools::writer(matches.free.get(1))?;
    let with_ids = matches.opt_present("with-ids");
    match mesh.dimension() {
        2 => write_points::<2>(output, &mesh, with_ids)?,
        3 => write_points::<3>(output, &mesh, with_ids)?,
        n => anyhow::bail!("expected 2D or 3D mesh, got a {n}D mesh"),
    };

    Ok(())
}
