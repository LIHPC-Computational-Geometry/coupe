use anyhow::Context as _;
use anyhow::Result;
use mesh_io::medit::Mesh;
use std::env;
use std::io;

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt(
        "n",
        "times",
        "numbers of times to refine (default: 1)",
        "NUMBER",
    );

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: mesh-refine [options]"));
        return Ok(());
    }

    let n: usize = matches
        .opt_get("n")
        .context("invalid value for option 'times'")?
        .unwrap_or(1);

    eprintln!("Reading mesh...");
    let stdin = io::stdin();
    let stdin = stdin.lock();
    let stdin = io::BufReader::new(stdin);
    let mut mesh = Mesh::from_reader(stdin).unwrap();
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
    println!("{}", mesh);

    Ok(())
}
