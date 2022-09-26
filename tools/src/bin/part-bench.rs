use anyhow::Context as _;
use anyhow::Result;
use coupe::nalgebra::allocator::Allocator;
use coupe::nalgebra::ArrayStorage;
use coupe::nalgebra::Const;
use coupe::nalgebra::DefaultAllocator;
use coupe::nalgebra::DimDiff;
use coupe::nalgebra::DimSub;
use coupe::nalgebra::ToTypenum;
use criterion::Criterion;
use mesh_io::weight;
use mesh_io::Mesh;
use std::env;
use std::fs;
use std::io;
use std::thread::sleep;
use std::time::Duration;

const USAGE: &str = "Usage: part-bench [options]";

fn criterion_options(options: &mut getopts::Options) {
    // TODO use Criterion::configure_with_args when it respects POSIX's "--"
    // TODO more options if needed
    options.optopt("b", "baseline", "Compare to a named baseline", "NAME");
    options.optopt(
        "s",
        "save-baseline",
        "Save results to a named baseline",
        "NAME",
    );
    options.optopt(
        "",
        "sample-size",
        "Changes the default size of the sample for this run (default: 100)",
        "NAME",
    );
}

fn configure_criterion(mut c: Criterion, matches: &getopts::Matches) -> Result<Criterion> {
    if let Some(baseline) = matches.opt_str("b") {
        c = c.retain_baseline(baseline);
    }
    if let Some(baseline) = matches.opt_str("s") {
        c = c.save_baseline(baseline);
    }
    if let Some(n) = matches.opt_get("sample-size")? {
        c = c.sample_size(n);
    }
    Ok(c)
}

fn build_pool(thread_count: usize) -> rayon::ThreadPool {
    let core_count = affinity::get_core_num();
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .spawn_handler(|thread| {
            let mut b = std::thread::Builder::new();
            if let Some(name) = thread.name() {
                b = b.name(name.to_owned());
            }
            if let Some(stack_size) = thread.stack_size() {
                b = b.stack_size(stack_size);
            }
            b.spawn(move || {
                let core_idx = thread.index() % core_count;
                affinity::set_thread_affinity([core_idx]).unwrap();
                thread.run();
            })?;
            Ok(())
        })
        .build()
        .unwrap()
}

fn main_d<const D: usize>(
    matches: getopts::Matches,
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
    let mut algorithms: Vec<_> = algorithm_specs
        .iter()
        .map(|algorithm_spec| {
            coupe_tools::parse_algorithm::<D>(algorithm_spec)
                .with_context(|| format!("invalid algorithm {:?}", algorithm_spec))
        })
        .collect::<Result<_>>()?;

    let mut partition = vec![0; coupe_tools::used_element_count(&mesh)];
    let problem = coupe_tools::Problem::new(mesh, weights, edge_weights);

    let intel_domain = coupe_tools::ittapi::domain("algorithm-chain");

    let mut runners: Vec<_> = algorithms
        .iter_mut()
        .zip(&algorithm_specs)
        .map(|(algorithm, algorithm_spec)| {
            let name = format!("{algorithm_spec}.to_runner");
            let _task = coupe_tools::ittapi::begin(&intel_domain, &name);

            algorithm.to_runner(&problem)
        })
        .collect();
    let mut benchmark = || {
        let _task = coupe_tools::ittapi::begin(&intel_domain, "benchmark-iteration");

        for runner in &mut runners {
            runner(&mut partition).unwrap();
        }
    };

    let mut c = configure_criterion(Criterion::default(), &matches)?.with_output_color(true);

    let benchmark_name = {
        use std::path::PathBuf;

        let mesh_file = matches.opt_str("m").unwrap();
        let mesh_file = PathBuf::from(mesh_file);
        let mesh_file = mesh_file.file_stem().unwrap().to_str().unwrap();

        let weight_file = matches.opt_str("w").unwrap();
        let weight_file = PathBuf::from(weight_file);
        let mut weight_file = weight_file.file_stem().unwrap().to_str().unwrap();

        if let Some(wf) = weight_file.strip_prefix(mesh_file) {
            weight_file = wf;
            if let Some(wf) = weight_file.strip_prefix('.') {
                weight_file = wf;
            }
        };
        if weight_file.is_empty() {
            format!("{mesh_file};{}", algorithm_specs.join(";"))
        } else {
            format!("{mesh_file};{weight_file};{}", algorithm_specs.join(";"))
        }
    };
    if matches.opt_present("e") {
        let max_threads = rayon::current_num_threads();
        let mut g = c.benchmark_group(benchmark_name);
        let mut thread_count = 1;
        loop {
            let pool = build_pool(thread_count);
            let benchmark_name = format!("threads={thread_count}");
            g.bench_function(&benchmark_name, |b| pool.install(|| b.iter(&mut benchmark)));
            thread_count *= 2;
            if max_threads < thread_count {
                break;
            }
            println!("Waiting 4s for CPUs to cool down...");
            sleep(Duration::from_secs(4));
        }
    } else {
        c.bench_function(&benchmark_name, |b| b.iter(&mut benchmark));
    }

    Ok(partition)
}

fn main() -> Result<()> {
    #[cfg(debug_assertions)]
    eprintln!("Warning: This is a debug build of part-bench, benchmarks will not reflect real-world performance.");

    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optmulti(
        "a",
        "algorithm",
        "name of the algorithm to run, see ALGORITHMS",
        "NAME",
    );
    options.optflag("e", "efficiency", "Benchmark efficiency");
    options.optopt(
        "E",
        "edge-weights",
        "Change how edge weights are set",
        "VARIANT",
    );
    options.optopt("m", "mesh", "mesh file", "FILE");
    options.optopt("w", "weights", "weight file", "FILE");
    criterion_options(&mut options);

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage(USAGE));
        return Ok(());
    }
    if !matches.free.is_empty() {
        anyhow::bail!("too many arguments\n\n{}", options.usage(USAGE));
    }

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

    println!(" -> Dimension: {}", mesh.dimension());
    println!(" -> Number of nodes: {}", mesh.node_count());
    println!(" -> Number of elements: {}", mesh.element_count());

    match mesh.dimension() {
        2 => main_d::<2>(matches, edge_weights, mesh, weights)?,
        3 => main_d::<3>(matches, edge_weights, mesh, weights)?,
        n => anyhow::bail!("expected 2D or 3D mesh, got a {n}D mesh"),
    };

    Ok(())
}
