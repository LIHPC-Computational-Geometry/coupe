use anyhow::Context as _;
use anyhow::Result;
use itertools::Itertools as _;
use std::env;
use std::io;

#[derive(Clone, Copy)]
#[repr(usize)]
enum Axis {
    X,
    Y,
    Z,
}

impl std::str::FromStr for Axis {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "0" | "x" | "X" => Self::X,
            "1" | "y" | "Y" => Self::Y,
            "2" | "z" | "Z" => Self::Z,
            _ => anyhow::bail!("expected 0/1/2/x/y/z"),
        })
    }
}

#[derive(Copy, Clone)]
enum Distribution {
    Constant(f64),
    Linear(Axis, f64, f64),
}

fn parse_distribution(definition: &str) -> Result<Distribution> {
    let mut args = definition.split(',');
    let name = args.next().context("empty definition")?;

    fn required<T>(maybe_arg: Option<Result<T>>) -> Result<T> {
        maybe_arg.context("not enough arguments")?
    }

    fn f64_arg(arg: Option<&str>) -> Option<Result<f64>> {
        arg.map(|arg| {
            let f = arg
                .parse::<f64>()
                .with_context(|| format!("arg {:?} is not a valid float", arg))?;
            if !f.is_finite() {
                anyhow::bail!("arg {:?} is not finite", arg);
            }
            Ok(f)
        })
    }

    fn axis_arg(arg: Option<&str>) -> Option<Result<Axis>> {
        Some(
            arg?.parse::<Axis>()
                .with_context(|| format!("arg {:?} is not a valid axis", arg)),
        )
    }

    Ok(match name {
        "constant" => {
            let value = required(f64_arg(args.next()))?;
            Distribution::Constant(value)
        }
        "linear" => {
            let axis = required(axis_arg(args.next()))?;
            let from = required(f64_arg(args.next()))?;
            let to = required(f64_arg(args.next()))?;
            Distribution::Linear(axis, from, to)
        }
        _ => anyhow::bail!("unknown distribution {:?}", name),
    })
}

fn apply_distribution(d: Distribution, points: &[Vec<f64>]) -> Box<dyn Fn(&[f64]) -> f64> {
    match d {
        Distribution::Constant(value) => Box::new(move |_coordinates| value),
        Distribution::Linear(axis, from, to) => {
            let (min, max) = points
                .iter()
                .map(|point| point[axis as usize])
                .minmax()
                .into_option()
                .unwrap();
            let alpha = (to - from) / (max - min);
            let beta = -min * alpha;
            Box::new(move |coordinates| f64::mul_add(coordinates[axis as usize], alpha, beta))
        }
    }
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optmulti(
        "d",
        "distribution",
        "definition of the weight distribution, see DISTRIBUTION",
        "DEFINITION",
    );
    options.optflag(
        "i",
        "integers",
        "generate integers instead of floating-point numbers",
    );

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("Usage: weight-gen [options]");
        eprintln!();
        eprint!("Generate a weight distribution from the given mesh and prints it to STDOUT.");
        eprintln!("{}", options.usage(""));
        eprint!(include_str!("help_after.txt"));
        return Ok(());
    }

    let distributions: Vec<_> = matches
        .opt_strs("d")
        .into_iter()
        .map(|spec| parse_distribution(&spec))
        .collect::<Result<_>>()?;
    if distributions.is_empty() {
        anyhow::bail!("missing required option 'distribution'");
    }

    eprintln!("Reading mesh from standard input...");

    let input = io::stdin();
    let input = input.lock();
    let input = io::BufReader::new(input);
    let mesh = mesh_io::medit::Mesh::from_reader(input).context("failed to read mesh")?;

    let points: Vec<_> = mesh
        .elements()
        .filter_map(|(element_type, nodes, _element_ref)| {
            if element_type.dimension() != mesh.dimension() {
                return None;
            }
            let mut barycentre = vec![0.0; mesh.dimension()];
            for node_idx in nodes {
                let node_coordinates = mesh.node(*node_idx);
                for (bc_coord, node_coord) in barycentre.iter_mut().zip(node_coordinates) {
                    *bc_coord += node_coord;
                }
            }
            for bc_coord in &mut barycentre {
                *bc_coord = *bc_coord / nodes.len() as f64;
            }
            Some(barycentre)
        })
        .collect();

    let distributions: Vec<_> = distributions
        .into_iter()
        .map(|distribution| apply_distribution(distribution, &points))
        .collect();

    let weights = points
        .iter()
        .map(|point| distributions.iter().map(|distribution| distribution(point)));

    eprintln!("Writing weight distributions to standard output...");

    let output = io::stdout();
    let output = output.lock();
    let output = io::BufWriter::new(output);
    mesh_io::weight::write_floats(output, weights).context("failed to write weight array")?;

    Ok(())
}
