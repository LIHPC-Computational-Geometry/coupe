fn main() {
    let mut adjacency = sprs::CsMat::empty(sprs::CSR, 15);
    adjacency.reserve_outer_dim(15);
    adjacency.insert(0, 1, 1.0);
    adjacency.insert(0, 5, 1.0);

    adjacency.insert(1, 0, 1.0);
    adjacency.insert(1, 2, 1.0);
    adjacency.insert(1, 6, 1.0);

    adjacency.insert(2, 1, 1.0);
    adjacency.insert(2, 3, 1.0);
    adjacency.insert(2, 7, 1.0);

    adjacency.insert(3, 2, 1.0);
    adjacency.insert(3, 4, 1.0);
    adjacency.insert(3, 8, 1.0);

    adjacency.insert(4, 3, 1.0);
    adjacency.insert(4, 9, 1.0);

    adjacency.insert(5, 0, 1.0);
    adjacency.insert(5, 6, 1.0);
    adjacency.insert(5, 10, 1.0);

    adjacency.insert(6, 1, 1.0);
    adjacency.insert(6, 5, 1.0);
    adjacency.insert(6, 7, 1.0);
    adjacency.insert(6, 11, 1.0);

    adjacency.insert(7, 2, 1.0);
    adjacency.insert(7, 6, 1.0);
    adjacency.insert(7, 8, 1.0);
    adjacency.insert(7, 12, 1.0);

    adjacency.insert(8, 3, 1.0);
    adjacency.insert(8, 7, 1.0);
    adjacency.insert(8, 9, 1.0);
    adjacency.insert(8, 13, 1.0);

    adjacency.insert(9, 4, 1.0);
    adjacency.insert(9, 8, 1.0);
    adjacency.insert(9, 14, 1.0);

    adjacency.insert(10, 5, 1.0);
    adjacency.insert(10, 11, 1.0);

    adjacency.insert(11, 6, 1.0);
    adjacency.insert(11, 10, 1.0);
    adjacency.insert(11, 12, 1.0);

    adjacency.insert(12, 7, 1.0);
    adjacency.insert(12, 11, 1.0);
    adjacency.insert(12, 13, 1.0);

    adjacency.insert(13, 8, 1.0);
    adjacency.insert(13, 12, 1.0);
    adjacency.insert(13, 14, 1.0);

    adjacency.insert(14, 9, 1.0);
    adjacency.insert(14, 13, 1.0);

    let (xadj, adjncy, _) = adjacency.into_raw_storage();
    println!("{xadj:?}");
    println!("{adjncy:?}");
}
