use anyhow::Context as _;
use anyhow::Result;
use mesh_io::medit::Mesh;
use std::env;
use std::fs;
use std::io;
use std::io::Write as _;

fn apply(mesh: &mut Mesh, weights: impl Iterator<Item = isize>) {
    let mesh_dimension = mesh.dimension();
    for ((element_type, _nodes, element_ref), weight) in mesh.elements_mut().zip(weights) {
        if element_type.dimension() != mesh_dimension {
            continue;
        }
        *element_ref = weight;
    }
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt("m", "mesh", "mesh file", "FILE");
    options.optopt("w", "weights", "weight file", "FILE");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: apply-weights [options]"));
        return Ok(());
    }

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

    let stdout = io::stdout();
    let stdout = stdout.lock();
    let mut stdout = io::BufWriter::new(stdout);
    write!(stdout, "{}", mesh).context("failed to write mesh")?;

    Ok(())
}
