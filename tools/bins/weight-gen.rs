use anyhow::Context as _;
use anyhow::Result;
use coupe::PointND;
use mesh_io::Mesh;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::cmp;

const USAGE: &str = "Usage: weight-gen [options] [in-mesh [out-weights]] <in.mesh >out.weights";

fn partial_cmp(a: &f64, b: &f64) -> cmp::Ordering {
    if a < b {
        cmp::Ordering::Less
    } else {
        cmp::Ordering::Greater
    }
}

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
struct Spike<const D: usize> {
    height: f64,
    position: PointND<D>,
}

#[derive(Clone)]
enum Distribution<const D: usize> {
    Constant(f64),
    Linear(Axis, f64, f64),
    Spike(Vec<Spike<D>>),
}

fn parse_distribution<const D: usize>(definition: &str) -> Result<Distribution<D>> {
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
        let arg = arg?;
        Some(
            arg.parse::<Axis>()
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
        "spike" => {
            let mut spikes = Vec::new();
            while let Some(height) = f64_arg(args.next()) {
                let height = height?;
                if height <= 0.0 {
                    anyhow::bail!(
                        "expected 'spike' height to be strictly positive, found {height}"
                    );
                }
                let mut position = [0.0; D];
                for position in &mut position {
                    *position = required(f64_arg(args.next()))?;
                }
                spikes.push(Spike {
                    height,
                    position: PointND::from(position),
                });
            }
            Distribution::Spike(spikes)
        }
        _ => anyhow::bail!("unknown distribution {:?}", name),
    })
}

fn apply_distribution<const D: usize>(
    d: Distribution<D>,
    points: &[PointND<D>],
) -> Box<dyn Fn(PointND<D>) -> f64> {
    match d {
        Distribution::Constant(value) => Box::new(move |_coordinates| value),
        Distribution::Linear(axis, from, to) => {
            let axis_iter = points.par_iter().map(|point| point[axis as usize]);
            let (min, max) = rayon::join(
                || axis_iter.clone().min_by(partial_cmp).unwrap(),
                || axis_iter.clone().max_by(partial_cmp).unwrap(),
            );
            let mut alpha = if max == min {
                0.0
            } else {
                (to - from) / (max - min)
            };
            while to - from < alpha * (max - min) {
                alpha = coupe::nextafter(alpha, f64::NEG_INFINITY);
            }
            Box::new(move |coordinates| f64::mul_add(coordinates[axis as usize] - min, alpha, from))
        }
        Distribution::Spike(mut spikes) => {
            for spike in &mut spikes {
                spike.height = f64::ln(spike.height);
            }
            Box::new(move |point| {
                spikes
                    .iter()
                    .map(|spike| {
                        let distance = (spike.position - point).norm();
                        f64::exp(spike.height - distance)
                    })
                    .sum()
            })
        }
    }
}

fn weight_gen<const D: usize>(
    mesh: Mesh,
    distributions: Vec<String>,
    matches: getopts::Matches,
) -> Result<()> {
    let distributions: Vec<Distribution<D>> = distributions
        .into_iter()
        .map(|spec| parse_distribution(&spec))
        .collect::<Result<_>>()?;

    let points = coupe_tools::barycentres::<D>(&mesh);

    let distributions: Vec<_> = distributions
        .into_iter()
        .map(|distribution| apply_distribution(distribution, &points))
        .collect();

    let weights = points.iter().map(|point| {
        distributions
            .iter()
            .map(|distribution| distribution(*point))
    });

    let output = coupe_tools::writer(matches.free.get(1))?;
    if matches.opt_present("i") {
        let weights = weights.map(|weight| weight.map(|criterion| criterion as i64));
        mesh_io::weight::write_integers(output, weights)
    } else {
        mesh_io::weight::write_floats(output, weights)
    }
    .context("failed to write weight array")
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
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

    let matches = coupe_tools::parse_args(options, USAGE, 2)?;

    let distributions = matches.opt_strs("d");
    if distributions.is_empty() {
        anyhow::bail!("missing required option 'distribution'");
    }

    let mesh = coupe_tools::read_mesh(matches.free.first())?;

    match mesh.dimension() {
        2 => weight_gen::<2>(mesh, distributions, matches),
        3 => weight_gen::<3>(mesh, distributions, matches),
        n => anyhow::bail!("expected 2D or 3D mesh, got a {n}D mesh"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coupe::Point2D;
    use proptest::collection::vec;
    use proptest::strategy::Strategy;

    /// Strategy for finite, non-NaN floats that are within reasonable bounds.
    fn float() -> impl Strategy<Value = f64> {
        -1e150..1e150
    }

    proptest::proptest!(
        #[test]
        fn linear_within_bounds(
            points in vec(
                float().prop_map(|a| Point2D::new(a, a)),
                2..200
            ),
        ) {
            const LOW: f64 = 0.0;
            const HIGH: f64 = 100.0;
            let dist = Distribution::Linear(Axis::X, LOW, HIGH);
            let dist = apply_distribution(dist, &points);
            for p in points {
                let weight = dist(p);
                proptest::prop_assert!(
                    (LOW..=HIGH).contains(&weight),
                    "point {p:?} has weight {weight} which is not in [{LOW}, {HIGH}]",
                );
            }
        }
    );
}
