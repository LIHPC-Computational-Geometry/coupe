extern crate coupe;
#[macro_use]
extern crate clap;
extern crate examples;

use clap::App;
use coupe::algorithms::k_means::simplified_k_means;
use coupe::geometry::Point2D;

fn main() {
    let yaml = load_yaml!("../../simplified_k_means.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let n_max_iter: isize = matches
        .value_of("n_max_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for n_max_iter");

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

    let points = examples::generator::cicrcle_uniform(n_points, Point2D::new(0., 0.), 1.)
        .into_iter()
        .map(|p| p * p.y)
        .collect::<Vec<_>>();

    let weights: Vec<f64> = points.iter().map(|_| 1.).collect();

    let (partition, _weights) =
        simplified_k_means(points, weights, n_partitions, imbalance_tol, n_max_iter);

    if !matches.is_present("quiet") {
        examples::plot_partition(partition)
    }
}
