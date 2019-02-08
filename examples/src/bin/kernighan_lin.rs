use coupe::geometry::Point2D;
use coupe::partition::Partition;
use coupe::ProcessUniqueId;
use coupe::TopologicPartitionImprover;
use sprs::CsMat;

fn main() {
    //    swap
    // 0  1  0  1
    // +--+--+--+
    // |  |  |  |
    // +--+--+--+
    // 0  0  1  1
    let points = vec![
        Point2D::new(0., 0.),
        Point2D::new(1., 0.),
        Point2D::new(2., 0.),
        Point2D::new(3., 0.),
        Point2D::new(0., 1.),
        Point2D::new(1., 1.),
        Point2D::new(2., 1.),
        Point2D::new(3., 1.),
    ];
    let id0 = ProcessUniqueId::new();
    let id1 = ProcessUniqueId::new();

    let ids = vec![id0, id0, id1, id1, id0, id1, id0, id1];
    let weights = vec![1.; 8];
    let mut adjacency = CsMat::empty(sprs::CSR, 8);
    adjacency.insert(0, 1, 1.);
    adjacency.insert(1, 2, 1.);
    adjacency.insert(2, 3, 1.);
    adjacency.insert(4, 5, 1.);
    adjacency.insert(5, 6, 1.);
    adjacency.insert(6, 7, 1.);
    adjacency.insert(0, 4, 1.);
    adjacency.insert(1, 5, 1.);
    adjacency.insert(2, 6, 1.);
    adjacency.insert(3, 7, 1.);

    // symmetry
    adjacency.insert(1, 0, 1.);
    adjacency.insert(2, 1, 1.);
    adjacency.insert(3, 2, 1.);
    adjacency.insert(5, 4, 1.);
    adjacency.insert(6, 5, 1.);
    adjacency.insert(7, 6, 1.);
    adjacency.insert(4, 0, 1.);
    adjacency.insert(5, 1, 1.);
    adjacency.insert(6, 2, 1.);
    adjacency.insert(7, 3, 1.);

    println!("adjacency = {:#?}", adjacency);

    let partition = Partition::from_ids(&points, &weights, ids);

    let partition = coupe::KernighanLin::new(1, 2.).improve_partition(partition, adjacency.view());
    let ids = partition.into_ids();
    println!("new partition = {:#?}", ids);

    let to_plot = points.into_iter().zip(ids).collect::<Vec<_>>();
    examples::plot_partition(to_plot);
}
