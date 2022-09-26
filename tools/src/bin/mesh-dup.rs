use anyhow::Context as _;
use anyhow::Result;
use mesh_io::Mesh;
use std::env;
use std::io;

const USAGE: &str = "Usage: mesh-dup [options] <in.mesh >out.mesh";

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt("f", "format", "output format", "EXT");
    options.optopt("n", "times", "numbers of duplicates (default: 2)", "NUMBER");

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

    let n: usize = matches
        .opt_get("n")
        .context("invalid value for option 'times'")?
        .unwrap_or(2);

    let stdin = io::stdin();
    let stdin = stdin.lock();
    let stdin = io::BufReader::new(stdin);
    let mesh = Mesh::from_reader(stdin).context("failed to read mesh")?;
    coupe_tools::write_mesh(&mesh.duplicate(n), format)?;

    Ok(())
}
