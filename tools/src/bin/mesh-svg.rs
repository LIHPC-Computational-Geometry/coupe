use anyhow::Result;
use coupe::sprs::CsMatView;
use coupe::Point2D;
use mesh_io::ElementType;
use mesh_io::Mesh;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::collections::HashSet;
use std::env;
use std::io;

const USAGE: &str = "Usage: mesh-svg [options] [in-mesh [out-svg]] <in.mesh >out.svg";

/// Returns the list of elements that are interesting.
fn elements(mesh: &Mesh) -> impl Iterator<Item = (ElementType, &[usize], isize)> {
    mesh.elements()
        .filter(|(el_type, _el_nodes, _el_ref)| el_type.dimension() == mesh.dimension())
}

/// Returns a function that looks up an element given its ID.
fn element_lookup<'a>(mesh: &'a Mesh) -> impl Fn(usize) -> (ElementType, &'a [usize], isize) + 'a {
    struct ElementChunk<'a> {
        start_idx: usize,
        node_per_element: usize,
        nodes: &'a [usize],
    }
    let topology: Vec<ElementChunk> = mesh
        .topology()
        .iter()
        .filter(|(el_type, _nodes, _refs)| el_type.dimension() == mesh.dimension())
        .scan(0, |start_idx, (el_type, nodes, _refs)| {
            let item = ElementChunk {
                start_idx: *start_idx,
                node_per_element: el_type.node_count(),
                nodes,
            };
            *start_idx += nodes.len() / item.node_per_element;
            Some(item)
        })
        .collect();
    move |e| {
        for (item, t) in topology.iter().zip(mesh.topology()) {
            let e = e - item.start_idx;
            let e_node = e * item.node_per_element;
            if e_node < item.nodes.len() {
                let el_type = t.0;
                let el_nodes = &item.nodes[e_node..e_node + item.node_per_element];
                let el_ref = t.2[e];
                return (el_type, el_nodes, el_ref);
            }
        }
        unreachable!();
    }
}

/// Adds an element and its neighbors that have the same ref as `path_ref` to
/// `el_set`.
fn add_element_and_neighbors_to_path<'a>(
    el_set: &mut HashSet<usize>,
    path_ref: isize,
    el: usize,
    element_fn: impl Fn(usize) -> (ElementType, &'a [usize], isize) + Copy,
    adjacency: CsMatView<f64>,
) {
    let mut queue = Vec::new();
    queue.push(el);
    while let Some(el) = queue.pop() {
        if !el_set.insert(el) {
            continue;
        }
        for (neighbor, _) in adjacency.outer_view(el).unwrap().iter() {
            let (_el_type, _el_nodes, el_ref) = element_fn(neighbor);
            if el_ref == path_ref {
                queue.push(neighbor);
            }
        }
    }
}

fn twobytwo(path: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    path.iter()
        .zip(path[1..].iter().chain(std::iter::once(&path[0])))
        .map(|(node1, node2)| (*node1, *node2))
}

/// Merge a collection of connected paths into several unconnected paths.
///
/// Input paths are unordered lists of nodes, output paths are ordered lists of
/// nodes.
fn merge_paths(mut paths: Vec<Vec<usize>>) -> impl Iterator<Item = Vec<usize>> {
    std::iter::from_fn(move || {
        let mut p = match paths.pop() {
            Some(v) => v,
            None => return None,
        };
        let mut p_end = *p.last().unwrap();

        let mut merged_path = Vec::new();
        merged_path.extend_from_slice(&p);

        loop {
            p = match paths.iter().position(|q| {
                let q_start = *q.first().unwrap();
                let q_end = *q.last().unwrap();
                p_end == q_start || p_end == q_end
            }) {
                Some(idx) => {
                    let mut q = paths.remove(idx);
                    let q_start = q[0];
                    if p_end == q_start {
                        merged_path.pop();
                    } else {
                        q.pop();
                        q.reverse();
                    }
                    q
                }
                None => break,
            };
            p_end = *p.last().unwrap();
            merged_path.extend_from_slice(&p);
        }

        Some(merged_path)
    })
}

/// Returns the coordinates from a path and removes redundant nodes.
fn path_to_coords(mesh: &Mesh, path: Vec<usize>) -> Vec<Point2D> {
    fn are_aligned(p1: Point2D, p2: Point2D, p3: Point2D) -> bool {
        const EPSILON: f64 = 1e-9;
        let alignment = if f64::abs(p1[0] - p3[0]) < EPSILON {
            p2[0] - p1[0]
        } else if f64::abs(p1[1] - p3[1]) < EPSILON {
            p2[1] - p1[1]
        } else {
            (p2[0] - p1[0]) / (p3[0] - p1[0]) - (p2[1] - p1[1]) / (p3[1] - p1[1])
        };
        f64::abs(alignment) < EPSILON
    }

    let mut path_coords: Vec<Point2D> = Vec::with_capacity(path.len());
    for node in path {
        let coords = mesh.node(node);
        let coords = Point2D::new(coords[0], coords[1]);
        let len = path_coords.len();
        if let (Some(last_coords), Some(laster_coords)) =
            (path_coords.get(len - 2), path_coords.last())
        {
            if are_aligned(*laster_coords, *last_coords, coords) {
                path_coords[len - 1] = coords;
                continue;
            }
        }
        path_coords.push(coords);
    }
    path_coords
}

/// Returns the path that goes around the given set of elements.
fn frontier<'a>(
    el_set: &HashSet<usize>,
    mesh: &Mesh,
    element_fn: impl Fn(usize) -> (ElementType, &'a [usize], isize),
    adjacency: CsMatView<f64>,
) -> Vec<Vec<Point2D>> {
    let path_set = el_set
        .iter()
        .filter(|el| {
            // Select only frontier nodes:
            let neighbors_in_set = adjacency
                .outer_view(**el)
                .unwrap()
                .iter()
                .filter(|(neighbor, _)| el_set.contains(neighbor))
                .count();
            let (el_type, _, _) = element_fn(**el);
            let neighbors_total = el_type.node_count();
            neighbors_in_set != neighbors_total
        })
        .flat_map(|el| {
            // Select only edges that are not shared with neighbors in el_set:
            let (_, el_nodes, _) = element_fn(*el);
            let edges = twobytwo(el_nodes)
                .filter(|(node1, node2)| {
                    let is_within_el_set = adjacency
                        .outer_view(*el)
                        .unwrap()
                        .iter()
                        .filter(|(neighbor, _)| el_set.contains(neighbor))
                        .any(|(neighbor, _)| {
                            let (_, neighbor_nodes, _) = element_fn(neighbor);
                            neighbor_nodes.contains(node1) && neighbor_nodes.contains(node2)
                        });
                    !is_within_el_set
                })
                .map(|(n1, n2)| vec![n1, n2])
                .collect();
            merge_paths(edges)
        })
        .collect();
    merge_paths(path_set)
        .map(|path| path_to_coords(mesh, path))
        .collect()
}

struct Path {
    nodes: Vec<Vec<Point2D>>,
    color: isize,
}

/// Returns the list of "blobs"/frontiers/paths of the mesh's elements that
/// share the same reference.
fn paths(mesh: &Mesh) -> impl Iterator<Item = Path> + '_ {
    let adjacency = coupe_tools::dual(mesh);
    let mut visited = HashSet::new();
    let element_fn = element_lookup(mesh);

    elements(mesh)
        .enumerate()
        .filter_map(move |(el, (_el_type, _el_nodes, el_ref))| {
            if visited.contains(&el) {
                return None;
            }

            let mut el_set = HashSet::new();
            add_element_and_neighbors_to_path(
                &mut el_set,
                el_ref,
                el,
                &element_fn,
                adjacency.view(),
            );

            let nodes = frontier(&el_set, mesh, &element_fn, adjacency.view());

            visited.extend(el_set);

            Some(Path {
                nodes,
                color: el_ref,
            })
        })
}

fn write_svg<W>(mut w: W, mesh: &Mesh) -> Result<()>
where
    W: io::Write,
{
    let coordinates = unsafe {
        std::slice::from_raw_parts(
            mesh.coordinates().as_ptr() as *const Point2D,
            mesh.node_count(),
        )
    };
    let bb = match coupe::BoundingBox::<2>::from_points(coordinates.par_iter().cloned()) {
        Some(v) => v,
        None => return Ok(()),
    };
    let xmin = bb.p_min[0];
    let xmax = bb.p_max[0];
    let ymin = bb.p_min[1];
    let ymax = bb.p_max[1];
    let width = xmax - xmin;
    let height = ymax - ymin;
    writeln!(
        w,
        r#"<svg viewBox="{xmin} {ymin} {width} {height}" xmlns="http://www.w3.org/2000/svg">"#,
    )?;

    let color = {
        let ref_count = 1 + elements(mesh)
            .map(|(_, _, el_ref)| el_ref)
            .max()
            .unwrap_or(0);
        move |el_ref| {
            let brightness = (el_ref as f64 / ref_count as f64 * 256.0) as isize;
            brightness << 16 | brightness << 8 | brightness
        }
    };
    for path in paths(mesh) {
        write!(
            w,
            "<path fill=\"#{:06x}\" fill-rule=\"evenodd\" d=\"",
            color(path.color),
        )?;
        for node_list in path.nodes {
            let (first_node, nodes) = node_list.split_first().unwrap();
            write!(w, " M{},{} L", first_node[0], ymax - first_node[1] + ymin)?;
            for coords in nodes {
                write!(w, "{},{} ", coords[0], ymax - coords[1] + ymin)?;
            }
            write!(w, "Z")?;
        }
        writeln!(w, "\"/>")?;
    }

    writeln!(w, "</svg>")?;

    Ok(())
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage(USAGE));
        return Ok(());
    }
    if matches.free.len() > 2 {
        anyhow::bail!("too many arguments\n\n{}", options.usage(USAGE));
    }

    let mesh = coupe_tools::read_mesh(matches.free.get(0))?;

    let output = coupe_tools::writer(matches.free.get(1))?;
    match mesh.dimension() {
        2 => write_svg(output, &mesh)?,
        n => anyhow::bail!("expected 2D mesh, got a {n}D mesh"),
    };

    Ok(())
}
