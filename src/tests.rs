use algorithms;
use geometry::Point2D;
use itertools::Itertools;

#[test]
fn test_rcb_basic() {
    let ids: Vec<usize> = (0..8).collect();
    let weights = vec![1.; 8];
    let coordinates = vec![
        Point2D::new(-1.3, 6.),
        Point2D::new(2., -4.),
        Point2D::new(1., 1.),
        Point2D::new(-3., -2.5),
        Point2D::new(-1.3, -0.3),
        Point2D::new(2., 1.),
        Point2D::new(-3., 1.),
        Point2D::new(1.3, -2.),
    ];

    let partition = algorithms::geometric::rcb(ids, weights, coordinates, 2)
        .into_iter()
        .sorted_by(|(id1, _), (id2, _)| id1.cmp(id2));

    assert_eq!(partition[0].1, partition[6].1);
    assert_eq!(partition[1].1, partition[7].1);
    assert_eq!(partition[2].1, partition[5].1);
    assert_eq!(partition[3].1, partition[4].1);

    let (p_id1, p_id2, p_id3, p_id4) = (
        partition[0].1,
        partition[1].1,
        partition[2].1,
        partition[3].1,
    );

    let p1 = partition.iter().filter(|(_, p_id)| *p_id == p_id1);
    let p2 = partition.iter().filter(|(_, p_id)| *p_id == p_id2);
    let p3 = partition.iter().filter(|(_, p_id)| *p_id == p_id3);
    let p4 = partition.iter().filter(|(_, p_id)| *p_id == p_id4);

    assert_eq!(p1.count(), 2);
    assert_eq!(p2.count(), 2);
    assert_eq!(p3.count(), 2);
    assert_eq!(p4.count(), 2);
}
