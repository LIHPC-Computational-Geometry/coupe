use anyhow::Context as _;
use anyhow::Result;
use std::env;

const USAGE: &str = "Usage: mesh-dup [options] [in-mesh [out-mesh]] <in.mesh >out.mesh";

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optflag("", "version", "print version information");
    options.optopt("f", "format", "output format", "EXT");
    options.optopt("n", "times", "numbers of duplicates (default: 2)", "NUMBER");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        println!("{}", options.usage(USAGE));
        return Ok(());
    }
    if matches.opt_present("version") {
        println!("mesh-dup version {}", env!("COUPE_VERSION"));
        return Ok(());
    }
    if matches.free.len() > 2 {
        anyhow::bail!("too many arguments\n\n{}", options.usage(USAGE));
    }

    let format = matches
        .opt_get("f")
        .context("invalid value for option 'format'")?;

    let n: usize = matches
        .opt_get("n")
        .context("invalid value for option 'times'")?
        .unwrap_or(2);

    let mesh = coupe_tools::read_mesh(matches.free.get(0))?;
    let mesh = mesh.duplicate(n);
    coupe_tools::write_mesh(&mesh, format, matches.free.get(1))?;

    Ok(())
}
