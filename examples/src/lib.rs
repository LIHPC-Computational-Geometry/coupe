use gnuplot::{Color, Figure};
use itertools::Itertools;
use rand::Rng;
use snowflake::ProcessUniqueId;
use sprs::CsMat;

use coupe::geometry::Point2D;
use mesh_io::medit::ElementType;
use mesh_io::medit::MeditMesh;

// generate adjacency matrix from Medit mesh
pub fn generate_connectivity_matrix_medit(mesh: &MeditMesh) -> CsMat<u32> {
    let (_el_type, triangles) = mesh
        .topology()
        .iter()
        // TODO: really needs to filter on triangles?
        .find(|(el_type, _elements)| el_type == &ElementType::Triangle)
        .unwrap();

    let num_triangles = triangles.len() / 3;
    debug_assert_eq!(triangles.len() % 3, 0);
    let mut ret = CsMat::zero((num_triangles, num_triangles));
    for (i, t1) in triangles.chunks_exact(3).enumerate() {
        for (j, t2) in triangles.chunks_exact(3).enumerate() {
            if i == j {
                continue;
            }
            let mut nodes = t1.to_vec();
            nodes.extend(t2.iter().cloned());
            nodes.sort_unstable();
            nodes.dedup();
            debug_assert_eq!(nodes.len() % 2, 0);
            let num_common_nodes = 3 - nodes.len() / 2;
            const MIN_COMMON: usize = 1;
            if MIN_COMMON <= num_common_nodes {
                ret.insert(i, j, 1);
            }
        }
    }

    ret
}

pub fn plot_partition(points: Vec<(Point2D, ProcessUniqueId)>) {
    let color_map = points
        .into_iter()
        .map(|(p, id)| (id, p))
        .into_group_map()
        .into_iter()
        .map(|(_, points)| (random_color_string(), points))
        .map(|(col, points)| {
            (
                col,
                points.iter().map(|p| p.x).collect::<Vec<_>>(),
                points.iter().map(|p| p.y).collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    let mut fg = Figure::new();
    {
        let axes = fg.axes2d();

        for (col, xs, ys) in color_map {
            axes.points(&xs, &ys, &[Color(&col)]);
        }
    }

    fg.show();
}

fn random_color_string() -> String {
    let mut rng = rand::thread_rng();
    let hex = rng.gen_range(0..0xfff_fff);
    format!("#{:x}", hex)
}

pub mod generator {
    use coupe::geometry::{Point2D, Point3D};
    use rand::{self, Rng};

    pub fn circle_uniform(num_points: usize, center: Point2D, radius: f64) -> Vec<Point2D> {
        let bb_p_min = Point2D::new(center.x - radius, center.y - radius);
        let bb_p_max = Point2D::new(center.x + radius, center.y + radius);

        let mut rng = rand::thread_rng();
        let mut num_points = num_points;
        let mut ret = Vec::with_capacity(num_points);
        while num_points > 0 {
            let p = Point2D::new(
                rng.gen_range(bb_p_min.x..bb_p_max.x),
                rng.gen_range(bb_p_min.y..bb_p_max.y),
            );
            if (center - p).norm() < radius {
                num_points -= 1;
                ret.push(p);
            }
        }
        ret
    }

    pub fn rectangle_uniform(
        num_points: usize,
        center: Point2D,
        width: f64,
        height: f64,
    ) -> Vec<Point2D> {
        let p_min = Point2D::new(center.x - width / 2., center.y - height / 2.);
        let p_max = Point2D::new(center.x + width / 2., center.y + height / 2.);

        let mut rng = rand::thread_rng();
        (0..num_points)
            .map(|_| {
                Point2D::new(
                    rng.gen_range(p_min.x..p_max.x),
                    rng.gen_range(p_min.y..p_max.y),
                )
            })
            .collect()
    }

    pub fn square_uniform(num_points: usize, center: Point2D, length: f64) -> Vec<Point2D> {
        rectangle_uniform(num_points, center, length, length)
    }

    pub fn box_uniform(num_points: usize, p_min: Point3D, p_max: Point3D) -> Vec<Point3D> {
        let mut rng = rand::thread_rng();
        (0..num_points)
            .map(|_| {
                Point3D::new(
                    rng.gen_range(p_min.x..p_max.x),
                    rng.gen_range(p_min.y..p_max.y),
                    rng.gen_range(p_min.z..p_max.z),
                )
            })
            .collect()
    }
}
