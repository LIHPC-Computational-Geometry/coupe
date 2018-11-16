extern crate clap;
extern crate coupe;
extern crate examples;
extern crate itertools;
extern crate rand;

use clap::load_yaml;
use clap::App;

use coupe::algorithms::multi_jagged::*;
use coupe::geometry::Point2D;

fn main() {
    let yaml = load_yaml!("../../multi_jagged.yml");
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

    let max_iter: usize = matches
        .value_of("max_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for max_iter");

    let weights = vec![1.; num_points];
    // let weights = (1..=num_points).map(|i| i as f64).collect::<Vec<f64>>();

    // let points = examples::generator::rectangle_uniform(num_points, Point2D::new(0., 0.), 4., 2.);
    let points = examples::generator::circle_uniform(num_points, Point2D::new(0., 0.), 1.);

    let now = std::time::Instant::now();
    let partition = multi_jagged_2d(&points, &weights, num_partitions, max_iter);
    let end = now.elapsed();
    println!("elapsed in multi-jagged: {:?}", end);

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
