# The coupe toolkit

This directory contains tools to work around coupe, and other partitioners. It
includes the following tools:

- mesh-io, a library used to encode and decode meshes in different formats,
- num-part, a framework to evaluate the quality of number partitioning
  algorithms specifically,
- in the `src/bin` directory, a collection of tools to partition meshes and
  evaluate mesh partitions:
    - weight-gen generates a distribution of cell weights for a mesh,
    - mesh-part runs a partitioner on a given mesh and weight distribution, then
      outputs a partition file,
    - part-bench runs criterion on given partitioners, meshes and weights,
    - part-info displays information about a partition, for a given mesh and
      weight distribution,
    - apply-part encodes a partition in a mesh file for visualization.
    - apply-weight encodes a weight distribution in a mesh file for
      visualization,
    - mesh-svg outputs an SVG given a mesh file, for use with the two above
      tools,
    - mesh-dup and mesh-refine increase the size of a mesh by either duplicating
      the vertices or splitting its elements into smaller ones, respectively,
    - mesh-reorder changes the order of mesh elements,
    - mesh-points extracts the cell centers of a given mesh.
- in the `report` directory, a collection of shell scripts that aggregate
  results into visual reports:
    - `imbedgecut` generates an SVG graph comparing the imbalance/edge-cut of
      various algorithms,
    - `quality` generates an HTML report of partitioning results for a given
      mesh directory,
    - `efficiency` generates a CSV file and a SVG graph showing the efficiency
      (strong scaling) of algorithms,
    - `efficiency-weak` is the weak-scaling equivalent of the above tool. It
      will also run the algorithm itself to overcome a limitation in part-bench.

## Building

[scdoc] is required to build the man pages.

For end users, a simple Makefile with basic options is provided:

```
make
sudo make install
```

Otherwise, these tools can be built with cargo:

```
cargo build --bins
```

C bindings to mesh-io can be built with the following command:

```
cargo build -p mesh-io-ffi
```

### Integration with other partitioners

The `mesh-part` and `part-bench` tools have optional support for [METIS] and
[SCOTCH] that is disabled by default.  To enable these features, use the
`--features` command-line flag as shown below. Note that these features require
a working clang install (version 5.0 or higher).

```sh
# Enable SCOTCH and METIS algorithms
cargo build --bins --features metis,scotch
```

### Integration with Intel performance tools

The `mesh-part` and `part-bench` tools can better integrate with Intel VTune and
Advisor through the use of *Instrumentation and Tracing Technology APIs*.

To enable this integration, do so through the `intel-perf` cargo feature:

```sh
cargo build --bins --feature intel-perf
```

When enabled, `mesh-part` and `part-bench` will wrap algorithm calls into
*tasks*. See [Intel's documentation][intel] on how to analyze them.

## Usage

See the man pages in the `doc/` directory.

For example, here is a quick walk-through:

```shell
# Cell weights increase linearly according to its position on the X axis.
weight-gen --distribution linear,x,0,100 <heart.mesh >heart.linear.weights

# Apply coupe's hilbert curve followed by the Fidducia-Mattheyses algorithm.
mesh-part --algorithm hilbert,3 \
          --algorithm fm \
          --mesh heart.mesh \
          --weights heart.linear.weights \
          >heart.linear.hilbert-fm.part

# Apply METIS' recursive bisection on the same mesh and weights.
mesh-part --algorithm metis:recursive,3
          --mesh heart.mesh \
          --weights heart.linear.weights \
          >heart.linear.metis.part

# Compare both partitions.
part-info --mesh heart.mesh --weights heart.linear.weights \
          --partition heart.linear.hilbert-fm.part
part-info --mesh heart.mesh --weights heart.linear.weights \
          --partition heart.linear.metis.part
```

### Usage with Docker

Here is a quick walk-through allowing you to launch a container with Coupe bins
and to interact with it from your host.

```shell
# Pull docker image.
docker pull ghcr.io/lihpc-computational-geometry/coupe:main

# Add "coupe" tag to the docker image.
docker image tag ghcr.io/lihpc-computational-geometry/coupe:main coupe

# Run the container while binding the current directory and the container.
# This will allow you to access inner generated data from your host.
# Warning: removing a binded file from within the container also removes
# it from the host.
docker container run -dit \
    --name coupe_c \
    --mount type=bind,source="$(pwd)",target=/coupe/shared \
    coupe

# Generate a linear weight distribution from an embedded mesh file.
# Note: option `--integers` can be used to generate integers instead of
# floating-point numbers.
docker exec coupe_c sh -c 'weight-gen --distribution linear,x,0,100 \
    <shared/sample.mesh \
    >shared/sample.linear.weights'

# Partition the previous mesh and the generated weight distribution into 2 parts.
# using the Recursive Coordinate Bisection (RCB) algorithm coupled with the
# Fidducia-Mattheyses algorithm.
docker exec coupe_c sh -c 'mesh-part \
    --algorithm rcb,1 \
    --algorithm fm \
    --mesh shared/sample.mesh \
    --weights shared/sample.linear.weights \
    >shared/sample.linear.rcb-fm.part'

# Merge partition file into MEDIT mesh file and converts it to .svg file.
docker exec coupe_c sh -c 'apply-part \
    --mesh shared/sample.mesh \
    --partition shared/sample.linear.rcb-fm.part \
    | mesh-svg >shared/sample.rcb-fm.svg'

# Open the svg file in firefox.
firefox sample.rcb-fm.svg &
```

[intel]: https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance/code-profiling-scenarios/task-analysis.html#task-analysis_TOP_TASKS
[METIS]: https://github.com/LIHPC-Computational-Geometry/metis-rs
[SCOTCH]: https://github.com/LIHPC-Computational-Geometry/scotch-rs
[scdoc]: https://sr.ht/~sircmpwn/scdoc/
