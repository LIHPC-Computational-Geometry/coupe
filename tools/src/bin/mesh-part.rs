use anyhow::Context as _;
use anyhow::Result;
use coupe::nalgebra::allocator::Allocator;
use coupe::nalgebra::ArrayStorage;
use coupe::nalgebra::Const;
use coupe::nalgebra::DefaultAllocator;
use coupe::nalgebra::DimDiff;
use coupe::nalgebra::DimSub;
use coupe::nalgebra::ToTypenum;
use mesh_io::weight;
use mesh_io::Mesh;
use std::env;
use std::fs;
use std::io;
use std::mem::drop;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::Registry;
use tracing_tree::HierarchicalLayer;

const USAGE: &str = "Usage: mesh-part [options] [out-part] >out.part";

fn main_d<const D: usize>(
    matches: &getopts::Matches,
    edge_weights: coupe_tools::EdgeWeightDistribution,
    mesh: Mesh,
    weights: weight::Array,
) -> Result<Vec<usize>>
where
    Const<D>: DimSub<Const<1>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
{
    let algorithm_specs = matches.opt_strs("a");
    let algorithms: Vec<_> = algorithm_specs
        .iter()
        .map(|algorithm_spec| {
            coupe_tools::parse_algorithm::<D>(algorithm_spec)
                .with_context(|| format!("invalid algorithm {:?}", algorithm_spec))
        })
        .collect::<Result<_>>()?;

    let mut partition = vec![0; coupe_tools::used_element_count(&mesh)];
    let problem = coupe_tools::Problem::new(mesh, weights, edge_weights);

    let show_metadata = matches.opt_present("v");

    let intel_domain = coupe_tools::ittapi::domain("algorithm-chain");

    for (algorithm_spec, mut algorithm) in algorithm_specs.iter().zip(algorithms) {
        let name = format!("{algorithm_spec}.to_runner");
        let task = coupe_tools::ittapi::begin(&intel_domain, &name);

        let mut algorithm = algorithm.to_runner(&problem);

        drop(task);
        let task = coupe_tools::ittapi::begin(&intel_domain, algorithm_spec);

        let metadata = algorithm(&mut partition)
            .with_context(|| format!("failed to apply algorithm {:?}", algorithm_spec))?;

        drop(task);

        if !show_metadata {
            continue;
        }
        if let Some(metadata) = metadata {
            eprintln!("{algorithm_spec}: {metadata:?}");
        } else {
            eprintln!("{algorithm_spec}:");
        }
    }

    Ok(partition)
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optmulti(
        "a",
        "algorithm",
        "name of the algorithm to run, see ALGORITHMS",
        "NAME",
    );
    options.optopt(
        "E",
        "edge-weights",
        "Change how edge weights are set",
        "VARIANT",
    );
    options.optopt("m", "mesh", "mesh file", "FILE");
    options.optopt("t", "trace", "emit a chrome trace", "FILE");
    options.optflag("v", "verbose", "print diagnostic data");
    options.optopt("w", "weights", "weight file", "FILE");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage(USAGE));
        return Ok(());
    }
    if matches.free.len() > 1 {
        anyhow::bail!("too many arguments\n\n{}", options.usage(USAGE));
    }

    let registry = Registry::default().with(EnvFilter::from_env("LOG")).with(
        HierarchicalLayer::new(4)
            .with_thread_ids(true)
            .with_targets(true)
            .with_bracketed_fields(true),
    );
    let _chrome_trace_guard = match matches.opt_str("t") {
        Some(filename) => {
            let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .file(filename)
                .build();
            registry.with(chrome_layer).init();
            Some(guard)
        }
        None => {
            registry.init();
            None
        }
    };

    let edge_weights = matches
        .opt_get("E")
        .context("invalid value for -E, --edge-weights")?
        .unwrap_or(coupe_tools::EdgeWeightDistribution::Uniform);

    let mesh_file = matches
        .opt_str("m")
        .context("missing required option 'mesh'")?;
    let mesh_file = fs::File::open(mesh_file).context("failed to open mesh file")?;
    let mesh_file = io::BufReader::new(mesh_file);

    let weight_file = matches
        .opt_str("w")
        .context("missing required option 'weights'")?;
    let weights = fs::File::open(&weight_file).context("failed to open weight file")?;
    let weights = io::BufReader::new(weights);

    let (mesh, weights) = rayon::join(
        || Mesh::from_reader(mesh_file).context("failed to read mesh file"),
        || weight::read(weights).context("failed to read weight file"),
    );
    let mesh = mesh?;
    let weights = weights?;

    let partition = match mesh.dimension() {
        2 => main_d::<2>(&matches, edge_weights, mesh, weights)?,
        3 => main_d::<3>(&matches, edge_weights, mesh, weights)?,
        n => anyhow::bail!("expected 2D or 3D mesh, got a {n}D mesh"),
    };

    let output = coupe_tools::writer(matches.free.get(1))?;
    mesh_io::partition::write(output, partition).context("failed to write partition")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use coupe::sprs::CsMat;
    use coupe::sprs::CSR;

    #[test]
    fn test_adjacency_convert() {
        let mut adjacency = CsMat::empty(CSR, 15);
        adjacency.reserve_outer_dim(15);
        adjacency.insert(0, 1, 1.0);
        adjacency.insert(0, 5, 1.0);

        adjacency.insert(1, 0, 1.0);
        adjacency.insert(1, 2, 1.0);
        adjacency.insert(1, 6, 1.0);

        adjacency.insert(2, 1, 1.0);
        adjacency.insert(2, 3, 1.0);
        adjacency.insert(2, 7, 1.0);

        adjacency.insert(3, 2, 1.0);
        adjacency.insert(3, 4, 1.0);
        adjacency.insert(3, 8, 1.0);

        adjacency.insert(4, 3, 1.0);
        adjacency.insert(4, 9, 1.0);

        adjacency.insert(5, 0, 1.0);
        adjacency.insert(5, 6, 1.0);
        adjacency.insert(5, 10, 1.0);

        adjacency.insert(6, 1, 1.0);
        adjacency.insert(6, 5, 1.0);
        adjacency.insert(6, 7, 1.0);
        adjacency.insert(6, 11, 1.0);

        adjacency.insert(7, 2, 1.0);
        adjacency.insert(7, 6, 1.0);
        adjacency.insert(7, 8, 1.0);
        adjacency.insert(7, 12, 1.0);

        adjacency.insert(8, 3, 1.0);
        adjacency.insert(8, 7, 1.0);
        adjacency.insert(8, 9, 1.0);
        adjacency.insert(8, 13, 1.0);

        adjacency.insert(9, 4, 1.0);
        adjacency.insert(9, 8, 1.0);
        adjacency.insert(9, 14, 1.0);

        adjacency.insert(10, 5, 1.0);
        adjacency.insert(10, 11, 1.0);

        adjacency.insert(11, 6, 1.0);
        adjacency.insert(11, 10, 1.0);
        adjacency.insert(11, 12, 1.0);

        adjacency.insert(12, 7, 1.0);
        adjacency.insert(12, 11, 1.0);
        adjacency.insert(12, 13, 1.0);

        adjacency.insert(13, 8, 1.0);
        adjacency.insert(13, 12, 1.0);
        adjacency.insert(13, 14, 1.0);

        adjacency.insert(14, 9, 1.0);
        adjacency.insert(14, 13, 1.0);

        let (xadj, adjncy, _) = adjacency.into_raw_storage();

        assert_eq!(
            xadj.as_slice(),
            &[0, 2, 5, 8, 11, 13, 16, 20, 24, 28, 31, 33, 36, 39, 42, 44]
        );
        assert_eq!(
            adjncy.as_slice(),
            &[
                1, 5, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 9, 0, 6, 10, 1, 5, 7, 11, 2, 6, 8, 12, 3, 7, 9,
                13, 4, 8, 14, 5, 11, 6, 10, 12, 7, 11, 13, 8, 12, 14, 9, 13
            ]
        );
    }
}
