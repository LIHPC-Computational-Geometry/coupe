extern crate clap;
extern crate coupe;
extern crate examples;

use clap::load_yaml;
use clap::App;
use coupe::algorithms::k_means::simplified_k_means;
use coupe::geometry::Point2D;

fn main() {
    let yaml = load_yaml!("../../simplified_k_means.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let max_iter: isize = matches
        .value_of("max_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for max_iter");

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

    let hilbert: bool = matches.is_present("hilbert");

    let points = examples::generator::circle_uniform(num_points, Point2D::new(0., 0.), 1.)
        .into_iter()
        // .map(|p| p * p.y)
        .collect::<Vec<_>>();

    // let points = examples::generator::rectangle_uniform(num_points, Point2D::new(0., 0.), 8., 4.);

    let weights: Vec<f64> = points.iter().map(|_| 1.).collect();

    let partition = simplified_k_means(
        &points,
        &weights,
        num_partitions,
        imbalance_tol,
        max_iter,
        hilbert,
    );

    if !matches.is_present("quiet") {
        let part = points.into_iter().zip(partition).collect::<Vec<_>>();
        examples::plot_partition(part)
    }
}
