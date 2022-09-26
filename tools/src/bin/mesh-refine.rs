use anyhow::Context as _;
use anyhow::Result;
use std::env;

const USAGE: &str = "Usage: mesh-refine [options] [in-mesh [out-mesh]] <in.mesh >out.mesh";

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt("f", "format", "output format", "EXT");
    options.optopt(
        "n",
        "times",
        "numbers of times to refine (default: 1)",
        "NUMBER",
    );

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage(USAGE));
        return Ok(());
    }
    if matches.free.len() > 2 {
        anyhow::bail!("too many arguments\n\n{}", options.usage(USAGE));
    }

    let format: coupe_tools::MeshFormat = matches
        .opt_get("f")
        .context("invalid value for option 'format'")?
        .unwrap_or(coupe_tools::MeshFormat::MeditBinary);

    let n: usize = matches
        .opt_get("n")
        .context("invalid value for option 'times'")?
        .unwrap_or(1);

    eprintln!("Reading mesh...");
    let mut mesh = coupe_tools::read_mesh(matches.free.get(0))?;
    eprintln!(" -> Dimension: {}", mesh.dimension());
    eprintln!(" -> Nodes: {}", mesh.node_count());
    eprintln!(" -> Elements: {}", mesh.element_count());

    for i in 0..n {
        eprint!("\rRefining mesh... {i:2}/{n}");
        mesh = mesh.refine();
    }
    eprintln!("\rRefining mesh... done    ");

    eprintln!(" -> Nodes: {}", mesh.node_count());
    eprintln!(" -> Elements: {}", mesh.element_count());

    eprintln!("Writing mesh...");
    coupe_tools::write_mesh(&mesh, format, matches.free.get(1))?;

    Ok(())
}
