# coupe

This is the main repository for coupe, a mesh partitioning library.

coupe currently implements the following geometric and topologic algorithms (read the API docs for more information):
  - RCB (Recursive Coordinate Bisection)
  - RIB (Recusrive Inertial Bisection)
  - Multi-Jagged
  - Z space filling curve
  - Hilbert space filling curve (2D only)
  - K-Means clustering algorithm
  - Greedy Graph Growing
  - Kernighan-Lin (bipartition only)
  - Fiduccia-Mattheyses

There are two different kinds of algorithms: those which generate a partition from scratch and those which
improve an existing partition. These kinds discrimination is enforced by traits implementation on algorithms structs.
The `Compose` trait can be used to compose algorithms to produce more complexe ones. For intance, we can compose 
a fast initial partitioning algorithm such as the Hilbert space filling curve with a heavier partition improving algorithm such as K-Means to produce
a new initial partitioning algorithm.

## Running examples
Several examples are located in the `examples` subcrate.

```shell
cargo run -p examples --bin <EXAMPLE_NAME> -- <EXAMPLE_ARGS>
```
The `mesh_file` example can run various algorithms by loading meshes from files, using `mesh_io`, whereas other examples
use randomly generated meshes.

## Usage

A minimal coupe usage example using the Recursive Coordinates Bisection algorithm.

```rust
use coupe::Partitioner;
use coupe::{Point2D, Rcb};
use rand::Rng;

const NUM_POINTS: usize = 5000;

fn main() {
    let mut rng = rand::thread_rng();

    // generate points sample
    let points: Vec<Point2D> = (0..NUM_POINTS)
        .map(|_| Point2D::new(rng.gen_range(0., 1.), rng.gen_range(0., 1.)))
        .collect();

    // generate weights sample
    let weights: Vec<f64> = (0..NUM_POINTS).map(|_| rng.gen_range(1., 3.)).collect();

    // Create an object that describes the partitioning
    // algorithm we want to use

    // here we use the RCB algorithm with 3 iterations
    // that will yield a partition of 2^3 = 8 parts
    let algo = Rcb::new(3);

    // run the algorithm on our points and weights
    // don't forget to import the `Partitioner` trait in scope
    let partition = algo.partition(points.as_slice(), &weights);

    // compute some metrics about the generated partition
    // first, compute the aspect ratios of each parts
    let aspect_ratios: Vec<f64> = partition
        .parts() // generates an iterator over every part
        .map(|part| part.aspect_ratio())
        .collect();

    // then compute the partition relative load imbalance
    // (which should be as low as possible)
    let relative_imbalance = partition.relative_imbalance();

    // display metrics
    println!("aspect ratios: {:?}", aspect_ratios);
    println!("relative imbalance: {}%", 100. * relative_imbalance); // displays in % of total weight

    // we can also extract raw ids array by consuming
    // the partition object
    let _ids = partition.into_ids();
}
```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

`SPDX-License-Identifier: Apache-2.0 OR MIT`

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.