use clap::load_yaml;
use clap::App;

use coupe::geometry::Point2D;
use coupe::HilbertCurve;
use coupe::Partitioner;

fn main() {
    let yaml = load_yaml!("../../hilbert_curve.yml");
    let matches = App::from_yaml(yaml).get_matches();

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

    let order: usize = matches
        .value_of("order")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for order");

    let hilbert_curve = HilbertCurve {
        num_partitions,
        order: order as u32,
    };

    let weights = vec![1.; num_points];

    let points = examples::generator::rectangle_uniform(num_points, Point2D::new(0., 0.), 4., 2.);
    // let points = examples::generator::circle_uniform(num_points, Point2D::new(0., 0.), 1.);

    let now = std::time::Instant::now();
    let partition = hilbert_curve.partition(&points, &weights);
    let end = now.elapsed();
    println!("elapsed in hilbert_curve_partition: {:?}", end);

    let max_imbalance = coupe::analysis::imbalance_max_diff(&weights, &partition);
    let relative_imbalance = coupe::analysis::imbalance_relative_diff(&weights, &partition);
    let mut aspect_ratios = coupe::analysis::aspect_ratios(&partition, &points)
        .into_iter()
        .map(|(_id, r)| r)
        .collect::<Vec<_>>();;
    aspect_ratios
        .as_mut_slice()
        .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    println!("Partition analysis:");
    println!("   > max weight diff: {}", max_imbalance);
    println!("   > relative weight diff: {}%", 100. * relative_imbalance);
    println!("   > ordered aspect ratios: {:?}", aspect_ratios);

    if !matches.is_present("quiet") {
        let part = points.into_iter().zip(partition).collect::<Vec<_>>();

        examples::plot_partition(part)
    }
}
