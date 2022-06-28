use std::error::Error;

use coupe::Partition as _;
use coupe::Point2D;

fn main() -> Result<(), Box<dyn Error>> {
    // Let's define a graph:
    //
    //     Node IDs:       Weights:
    //
    //     2    5    8     3    4    5
    //      +---+---+       +---+---+
    //      |   |   |       |   |   |
    //     1+---4---+7     2+---3---+4
    //      |   |   |       |   |   |
    //      +---+---+       +---+---+
    //     0    3    6     1    2    3
    //
    let coordinates: [Point2D; 9] = [
        Point2D::new(0.0, 0.0),
        Point2D::new(0.0, 1.0),
        Point2D::new(0.0, 2.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(1.0, 1.0),
        Point2D::new(1.0, 2.0),
        Point2D::new(2.0, 0.0),
        Point2D::new(2.0, 1.0),
        Point2D::new(2.0, 2.0),
    ];

    let weights: [f64; 9] = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0];

    let graph: sprs::CsMat<i64> = {
        let mut g = sprs::CsMat::empty(sprs::CSR, 9);
        g.insert(0, 1, 1);
        g.insert(0, 3, 1);

        g.insert(1, 0, 1);
        g.insert(1, 2, 1);
        g.insert(1, 4, 1);

        g.insert(2, 1, 1);
        g.insert(2, 5, 1);

        g.insert(3, 0, 1);
        g.insert(3, 4, 1);
        g.insert(3, 6, 1);

        g.insert(4, 1, 1);
        g.insert(4, 3, 1);
        g.insert(4, 5, 1);
        g.insert(4, 7, 1);

        g.insert(5, 2, 1);
        g.insert(5, 4, 1);
        g.insert(5, 8, 1);

        g.insert(6, 3, 1);
        g.insert(6, 7, 1);

        g.insert(7, 4, 1);
        g.insert(7, 6, 1);
        g.insert(7, 8, 1);

        g.insert(8, 5, 1);
        g.insert(8, 7, 1);

        g
    };

    let mut partition = [0; 9];

    fn print_partition(partition: &[usize]) {
        println!();
        println!("  {}---{}---{}", partition[2], partition[5], partition[8]);
        println!("  |   |   |");
        println!("  {}---{}---{}", partition[1], partition[4], partition[7]);
        println!("  |   |   |");
        println!("  {}---{}---{}", partition[0], partition[3], partition[6]);
        println!();
    }

    coupe::HilbertCurve {
        part_count: 2,
        ..Default::default()
    }
    .partition(&mut partition, (coordinates, weights))?;

    println!("Initial partitioning with a Hilbert curve:");
    print_partition(&partition);

    coupe::FiducciaMattheyses::default().partition(&mut partition, (graph.view(), &weights))?;

    println!("Partition improving with Fiduccia-Mattheyses:");
    print_partition(&partition);

    Ok(())
}
