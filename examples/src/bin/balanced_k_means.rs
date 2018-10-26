extern crate coupe;
#[macro_use]
extern crate clap;
extern crate examples;
extern crate itertools;

use clap::App;
use coupe::algorithms::k_means::{balanced_k_means, BalancedKmeansSettings};
use coupe::geometry::Point2D;

fn main() {
    let yaml = load_yaml!("../../balanced_k_means.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let n_max_iter: usize = matches
        .value_of("n_max_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for n_max_iter");

    let n_max_balance_iter: usize = matches
        .value_of("n_max_balance_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for n_max_balance_iter");

    let n_points: usize = matches
        .value_of("n_points")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for n_points");

    let n_partitions: usize = matches
        .value_of("n_partitions")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for n_partitions");

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

    let errode = matches.is_present("errode");

    let points = examples::generator::rectangle_uniform(n_points, Point2D::new(0., 0.), 4., 2.);

    let _weights: Vec<f64> = points.iter().map(|_| 1.).collect();

    let settings = BalancedKmeansSettings {
        num_partitions: n_partitions,
        imbalance_tol: imbalance_tol,
        max_iter: n_max_iter,
        max_balance_iter: n_max_balance_iter,
        delta_threshold: delta_max,
        errode: errode,
        ..Default::default()
    };

    let partition = balanced_k_means(points, settings);

    if !matches.is_present("quiet") {
        examples::plot_partition(partition)
    }
}
