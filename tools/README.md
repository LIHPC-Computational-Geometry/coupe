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
    - part-info displays information about a partition, for a given mesh and
      weight distribution,
    - apply-part encodes a partition in a mesh file for visualization.

## Building

These tools can be built with cargo:

```
cargo build --bins
```

The `mesh-part` tool has optional support for [MeTiS] and [SCOTCH]:

```
cargo build --bins --features metis,scotch
```

[MeTiS]: https://github.com/LIHPC-Computational-Geometry/metis-rs
[SCOTCH]: https://github.com/LIHPC-Computational-Geometry/scotch-rs

## Usage

Use the `--help` flag on any executable to open its manual.

For example,

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
