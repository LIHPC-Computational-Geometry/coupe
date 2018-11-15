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

    // let points = examples::generator::rectangle_uniform(num_points, Point2D::new(0., 0.), 4., 2.);
    let points = examples::generator::circle_uniform(num_points, Point2D::new(0., 0.), 1.);

    let now = std::time::Instant::now();
    let partition = multi_jagged_2d(&points, &weights, num_partitions, max_iter);
    let end = now.elapsed();
    println!("elapsed in multi-jagged: {:?}", end);

    let part_weights = coupe::analysis::weights(&weights, &partition)
        .into_iter()
        .map(|(_id, w)| w)
        .collect::<Vec<_>>();
    let imbalance = coupe::analysis::imbalance_max_diff(&weights, &partition);

    println!("Partition analysis:");
    println!("   > max weight diff: {}", imbalance);
    println!("   > parts weights: {:?}", part_weights);

    if !matches.is_present("quiet") {
        let part = points.into_iter().zip(partition).collect::<Vec<_>>();

        examples::plot_partition(part)
    }
}
