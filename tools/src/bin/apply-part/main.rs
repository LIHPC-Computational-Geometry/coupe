use anyhow::Context as _;
use anyhow::Result;
use std::env;
use std::fs;
use std::io;

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt("f", "format", "output format", "EXT");
    options.optopt("m", "mesh", "mesh file", "FILE");
    options.optopt("p", "partition", "partition file", "FILE");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: apply-part [options]"));
        return Ok(());
    }

    let format: coupe_tools::MeshFormat = matches
        .opt_get("f")
        .context("invalid value for option 'format'")?
        .unwrap_or(coupe_tools::MeshFormat::MeditBinary);

    let mesh_file = matches
        .opt_str("m")
        .context("missing required option 'mesh'")?;
    let mut mesh =
        mesh_io::medit::Mesh::from_file(mesh_file).context("failed to read mesh file")?;

    let partition_file = matches
        .opt_str("p")
        .context("missing required option 'partition'")?;
    let partition_file = fs::File::open(partition_file).context("failed to open partition file")?;
    let partition_file = io::BufReader::new(partition_file);
    let parts =
        mesh_io::partition::read(partition_file).context("failed to read partition file")?;

    let mesh_dimension = mesh.dimension();
    mesh.elements_mut()
        .filter(|(element_type, _, _)| element_type.dimension() == mesh_dimension)
        .zip(parts)
        .for_each(|((_, _, element_ref), part)| *element_ref = part as isize);

    coupe_tools::write_mesh(&mesh, format)?;

    Ok(())
}
