extern crate clap;
extern crate coupe;
extern crate examples;
extern crate itertools;

use clap::load_yaml;
use clap::App;
use coupe::algorithms::k_means::{balanced_k_means, BalancedKmeansSettings};
use coupe::geometry::Point2D;

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
    let hilbert = matches.is_present("hilbert");

    let points = examples::generator::rectangle_uniform(num_points, Point2D::new(0., 0.), 4., 2.);

    let weights = points
        .iter()
        .map(|p| if p.x < 0. { 1. } else { 2. })
        .collect::<Vec<_>>();

    let settings = BalancedKmeansSettings {
        num_partitions,
        imbalance_tol: imbalance_tol,
        max_iter,
        max_balance_iter,
        delta_threshold: delta_max,
        erode: erode,
        hilbert: hilbert,
        ..Default::default()
    };

    let (partition, _weights) = balanced_k_means(points, weights.clone(), settings);

    if !matches.is_present("quiet") {
        examples::plot_partition(partition)
    }
}
