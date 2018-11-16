extern crate clap;
extern crate coupe;
extern crate examples;
extern crate itertools;
extern crate rand;

use clap::load_yaml;
use clap::App;

use coupe::algorithms::recursive_bisection::rib_2d;
use coupe::geometry::Point2D;

fn main() {
    let yaml = load_yaml!("../../rcb.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let num_iter: usize = matches
        .value_of("num_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for num_iter");

    let num_points: usize = matches
        .value_of("num_points")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for num_points");

    let weights = vec![1.; num_points];

    let points = examples::generator::circle_uniform(num_points, Point2D::new(0., 0.), 1.)
        .into_iter()
        .map(|p| p * p.y)
        .collect::<Vec<_>>();

    let partition = rib_2d(&weights, &points.clone(), num_iter);

    if !matches.is_present("quiet") {
        let part = points.into_iter().zip(partition).collect::<Vec<_>>();
        examples::plot_partition(part)
    }
}
