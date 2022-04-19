# The coupe toolkit

This directory contains tools to work around coupe, and other partitioners. It
includes the following tools:

- mesh-io, a library used to encode and decode meshes in different formats,
- num-part, a framework to evaluate the quality of number partitioning
  algorithms specifically, and
- in the `src/bin` directory, a collection of tools to partition meshes and
  evaluate mesh partitions:
    - weight-gen generates a distribution of cell weights for a mesh,
    - mesh-part runs a partitioner on a given mesh and weight distribution,
    - part-bench runs criterion on given partitioners, meshes and weights,
    - part-info displays information about a partition, for a given mesh and
      weight distribution,
    - apply-part encodes a partition in a mesh file for visualization.
    - apply-weight encodes a weight distribution in a mesh file for
      visualization.
- in the `report` directory, a collection of shell scripts that aggregate
  results into visual reports:
    - `quality` generates an HTML report of partitioning results for a given
      mesh directory,
    - `efficiency` generates a CSV file and a SVG graph showing the efficiency
      (strong scaling) of algorithms.

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

The `mesh-part` and `part-bench` tools have optional support for [MeTiS] and
[SCOTCH] that is enabled by default.  To disable these features, use the
`--no-default-features` command-line flag.

```
# Disable SCOTCH and MeTiS support
cargo build --bins --no-default-features

# Enable MeTiS support only
cargo build --bins --no-default-features --features metis
```

C bindings to mesh-io can be built with the following command:

```
cargo build -p mesh-io-ffi
```

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

# Apply MeTiS' recursive bisection on the same mesh and weights.
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

[MeTiS]: https://github.com/LIHPC-Computational-Geometry/metis-rs
[SCOTCH]: https://github.com/LIHPC-Computational-Geometry/scotch-rs
[scdoc]: https://sr.ht/~sircmpwn/scdoc/
