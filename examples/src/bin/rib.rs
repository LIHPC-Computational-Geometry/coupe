extern crate coupe;
#[macro_use]
extern crate clap;
extern crate examples;
extern crate itertools;
extern crate rand;

use clap::App;
use itertools::Itertools;

use coupe::algorithms::geometric::rib;
use coupe::geometry::Point2D;

fn main() {
    let yaml = load_yaml!("../../rcb.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let n_iter: usize = matches
        .value_of("n_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for n_iter");

    let n_points: usize = matches
        .value_of("n_points")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for n_points");

    let ids: Vec<usize> = (0..n_points).collect();
    let weights = vec![1.; n_points];

    let points = examples::generator::cicrcle_uniform(n_points, Point2D::new(0., 0.), 1.)
        .into_iter()
        .map(|p| p * p.y)
        .collect::<Vec<_>>();

    let (partition, _weights, _points) = rib(ids, weights, points.clone(), n_iter);

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
