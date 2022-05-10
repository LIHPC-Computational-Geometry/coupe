use anyhow::Context as _;
use anyhow::Result;
use mesh_io::medit::Mesh;
use std::env;
use std::fs;
use std::io;

const USAGE: &str = "Usage: apply-weight [options] >out.mesh";

fn apply(mesh: &mut Mesh, weights: impl Iterator<Item = isize>) {
    let mesh_dimension = mesh.dimension();
    mesh.elements_mut()
        .filter(|(element_type, _, _)| element_type.dimension() == mesh_dimension)
        .zip(weights)
        .for_each(|((_, _, element_ref), weight)| *element_ref = weight);
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt("f", "format", "output format", "EXT");
    options.optopt("m", "mesh", "mesh file", "FILE");
    options.optopt("w", "weights", "weight file", "FILE");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage(USAGE));
        return Ok(());
    }
    if !matches.free.is_empty() {
        anyhow::bail!("too many arguments\n\n{}", options.usage(USAGE));
    }

    let format: coupe_tools::MeshFormat = matches
        .opt_get("f")
        .context("invalid value for option 'format'")?
        .unwrap_or(coupe_tools::MeshFormat::MeditBinary);

    let mesh_file = matches
        .opt_str("m")
        .context("missing required option 'mesh'")?;
    let mut mesh = Mesh::from_file(mesh_file).context("failed to read mesh file")?;

    let weight_file = matches
        .opt_str("w")
        .context("missing required option 'weights'")?;
    let weight_file = fs::File::open(weight_file).context("failed to open weight file")?;
    let weight_file = io::BufReader::new(weight_file);
    let weights = mesh_io::weight::read(weight_file).context("failed to read weight file")?;

    match weights {
        mesh_io::weight::Array::Integers(is) => {
            apply(&mut mesh, is.into_iter().map(|i| i[0] as isize));
        }
        mesh_io::weight::Array::Floats(fs) => {
            apply(&mut mesh, fs.into_iter().map(|f| f[0] as isize));
        }
    }

    coupe_tools::write_mesh(&mesh, format)?;

    Ok(())
}
