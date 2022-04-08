use anyhow::Context as _;
use anyhow::Result;
use criterion::Criterion;
use mesh_io::medit::Mesh;
use mesh_io::weight;
use std::env;
use std::fs;
use std::io;

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
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .unwrap()
}

fn main_d<const D: usize>(
    matches: getopts::Matches,
    mesh: Mesh,
    weights: weight::Array,
) -> Result<Vec<usize>> {
    let problem = coupe_tools::Problem {
        points: coupe_tools::barycentres::<D>(&mesh),
        weights,
        adjacency: coupe_tools::dual(&mesh),
    };
    let mut partition = vec![0; problem.points.len()];

    let algorithm_specs = matches.opt_strs("a");
    let mut algorithms: Vec<_> = algorithm_specs
        .iter()
        .map(|algorithm_spec| {
            coupe_tools::parse_algorithm(algorithm_spec)
                .with_context(|| format!("invalid algorithm {:?}", algorithm_spec))
        })
        .collect::<Result<_>>()?;

    let mut runners: Vec<_> = algorithms
        .iter_mut()
        .map(|algorithm| algorithm.to_runner(&problem))
        .collect();
    let mut benchmark = || {
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
        let weight_file = weight_file.file_stem().unwrap().to_str().unwrap();

        format!("{mesh_file}:{weight_file}:{}", algorithm_specs.join(":"))
    };
    if matches.opt_present("e") {
        let max_threads = matches.opt_get("e")?.unwrap_or_else(|| {
            let t = rayon::current_num_threads();
            println!("Number of threads available: {t}");
            t
        });
        let mut g = c.benchmark_group(benchmark_name);
        let mut thread_count = 1;
        while thread_count <= max_threads {
            let pool = build_pool(thread_count);
            let benchmark_name = format!("threads={thread_count}");
            g.bench_function(&benchmark_name, |b| pool.install(|| b.iter(&mut benchmark)));
            thread_count *= 2;
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
    options.optflagopt("e", "efficiency", "Benchmark efficiency", "MAX_THREADS");
    options.optopt("m", "mesh", "mesh file", "FILE");
    options.optopt("w", "weights", "weight file", "FILE");
    criterion_options(&mut options);

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: part-bench [options]"));
        return Ok(());
    }

    let mesh_file = matches
        .opt_str("m")
        .context("missing required option 'mesh'")?;
    println!("Reading {mesh_file:?}...");
    let mesh = Mesh::from_file(&mesh_file).context("failed to read mesh file")?;
    println!(" -> Dimension: {}", mesh.dimension());
    println!(" -> Number of nodes: {}", mesh.node_count());
    println!(" -> Number of elements: {}", mesh.element_count());
    println!();

    let weight_file = matches
        .opt_str("w")
        .context("missing required option 'weights'")?;
    println!("Reading {weight_file:?}...");
    let weights = fs::File::open(&weight_file).context("failed to open weight file")?;
    let weights = io::BufReader::new(weights);
    let weights = weight::read(weights).context("failed to read weight file")?;

    match mesh.dimension() {
        2 => main_d::<2>(matches, mesh, weights)?,
        3 => main_d::<3>(matches, mesh, weights)?,
        n => anyhow::bail!("expected 2D or 3D mesh, got a {n}D mesh"),
    };

    Ok(())
}
