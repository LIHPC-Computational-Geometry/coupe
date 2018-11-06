extern crate clap;
extern crate coupe;
extern crate examples;
extern crate itertools;
extern crate rand;

use clap::load_yaml;
use clap::App;
use itertools::Itertools;

use coupe::algorithms::geometric::rib;
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

    let ids: Vec<usize> = (0..num_points).collect();
    let weights = vec![1.; num_points];

    let points = examples::generator::cicrcle_uniform(num_points, Point2D::new(0., 0.), 1.)
        .into_iter()
        .map(|p| p * p.y)
        .collect::<Vec<_>>();

    let (partition, _weights, _points) = rib(ids, weights, points.clone(), num_iter);

    // sort ids
    let sorted_part_ids = partition
        .iter()
        .sorted_by(|(id1, _), (id2, _)| id1.cmp(id2));

    let points = points
        .into_iter()
        .zip(sorted_part_ids.iter().map(|(_, pid)| *pid))
        .collect::<Vec<_>>();

    if !matches.is_present("quiet") {
        examples::plot_partition(points)
    }
}
