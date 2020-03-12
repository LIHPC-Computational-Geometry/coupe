use clap::load_yaml;
use clap::App;
use coupe::geometry::Point2D;
use coupe::{Compose, Partitioner};
use coupe::{HilbertCurve, KMeans, KMeansBuilder};

fn main() {
    let yaml = load_yaml!("../../balanced_k_means.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let max_iter: usize = matches
        .value_of("max_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for max_iter");

    let max_balance_iter: usize = matches
        .value_of("max_balance_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for max_balance_iter");

    let num_points: usize = matches
        .value_of("num_points")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for num_points");

    let num_partitions: usize = matches
        .value_of("num_partitions")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for num_partitions");

    let imbalance_tol: f64 = matches
        .value_of("imbalance_tol")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for imbalance_tol");

    let delta_max: f64 = matches
        .value_of("delta_max")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for delta_max");

    let erode = matches.is_present("erode");

    let algo = HilbertCurve::new(num_partitions, 14).compose(
        KMeansBuilder::default()
            .num_partitions(num_partitions)
            .imbalance_tol(imbalance_tol)
            .max_iter(max_iter)
            .max_balance_iter(max_balance_iter)
            .delta_threshold(delta_max)
            .erode(erode)
            .build(),
    );

    let points = examples::generator::rectangle_uniform(num_points, Point2D::new(0., 0.), 4., 2.);

    let weights = points
        .iter()
        // .map(|p| if p.x < 0. { 1. } else { 2. })
        .map(|_p| 1.)
        .collect::<Vec<_>>();

    let now = std::time::Instant::now();
    let partition = algo.partition(points.as_slice(), &weights);
    let end = now.elapsed();
    println!("elapsed in balanced_k_means: {:?}", end);

    let max_imbalance = partition.max_imbalance();
    let relative_imbalance = partition.relative_imbalance();
    let mut aspect_ratios = partition
        .parts()
        .map(|part| part.aspect_ratio())
        .collect::<Vec<_>>();
    aspect_ratios
        .as_mut_slice()
        .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    println!("Partition analysis:");
    println!("   > max weight diff: {}", max_imbalance);
    println!("   > relative weight diff: {}%", 100. * relative_imbalance);
    println!("   > ordered aspect ratios: {:?}", aspect_ratios);

    if !matches.is_present("quiet") {
        let ids = partition.into_ids();
        let part = points.into_iter().zip(ids).collect::<Vec<_>>();
        examples::plot_partition(part)
    }
}
