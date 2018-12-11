extern crate clap;
extern crate coupe;
extern crate examples;
extern crate itertools;
extern crate rand;

use clap::load_yaml;
use clap::App;

use coupe::geometry::Point3D;
use coupe::Partitioner;
use coupe::Rib;

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

    let rib = Rib { num_iter };

    let weights = vec![1.; num_points];

    let mut points = examples::generator::box_uniform(
        num_points,
        Point3D::new(0., 0., 0.),
        Point3D::new(3., 2., 4.),
    );

    // dummy transform
    let axis = Point3D::new(1., 1., 1.);
    let mat = coupe::geometry::householder_reflection(&axis);
    let transform = |p| mat * p;
    for p in points.iter_mut() {
        *p = transform(*p);
    }

    let _partition = rib.partition(&points, &weights);

    if !matches.is_present("quiet") {
        unimplemented!("3D plot is not supported yet")
    }
}
