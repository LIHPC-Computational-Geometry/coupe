extern crate clap;
extern crate coupe;
extern crate examples;
extern crate itertools;
extern crate rand;

use clap::load_yaml;
use clap::App;

use coupe::algorithms::multi_jagged::multi_jagged_2d_with_scheme;
use coupe::geometry::Point2D;

fn main() {
    let yaml = load_yaml!("../../multi_jagged.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let num_points: usize = matches
        .value_of("num_points")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for num_points");

    let weights = vec![1.; num_points];

    let points = examples::generator::rectangle_uniform(num_points, Point2D::new(0., 0.), 4., 2.);

    let partition_scheme = vec![3, 3, 3];

    let now = std::time::Instant::now();
    let partition = multi_jagged_2d_with_scheme(&points, &weights, &partition_scheme);
    let end = now.elapsed();
    println!("elapsed in multi-jagged: {:?}", end);

    if !matches.is_present("quiet") {
        let part = points.into_iter().zip(partition).collect::<Vec<_>>();

        examples::plot_partition(part)
    }
}
