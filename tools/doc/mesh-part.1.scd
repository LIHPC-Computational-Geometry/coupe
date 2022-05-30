mesh-part(1)

# NAME

mesh-part - Partition a mesh

part-bench - Benchmark an algorithm

# SYNOPSIS

*mesh-part* [options...] >output.part

*part-bench* [options...]

# DESCRIPTION

mesh-part applies partitioning algorithms onto a given mesh in order, and writes
the resulting partition on standard output.

part-bench benchmarks the speed of the given algorithms and prints the results.

Only MEDIT meshes are supported, for now.  See *apply-part(1)*'s *INPUT FORMAT*
for details.

See *ALGORITHMS* for a list of supported algorithms.

See *PARTITION FILE* for a description of the output format of mesh-part.

# OPTIONS

*-h, --help*
	Show a help message and exit.

*-a, --algorithm* <spec>
	Apply the given algorithm on the mesh.  This option can be specified
	multiple times, so that algorithms can be chained together.  See
	*ALGORITHMS* for information on the _spec_ argument and a list of supported
	algorithms.

*-E, --edge-weights* <variant>
	Change how edge weights are set.  Possible values are:

	- _uniform_ (default): all edges have the same weight,
	- _linear_: edge weights are the sum of the vertex weights,
	- _sqrt_: edge weights are the sum of the vertex weights' square roots.

*-m, --mesh* <path>
	Required.  Partition the given mesh file.

*-w --weights* <path>
	Use the given weight file.  This file is expected to come from
	*weight-gen*(1) with the same mesh.

Options specific to *mesh-part*:

*-t, --trace* <path>
	Emits a trace file viewable in <chrome://tracing> or any compatible viewer.
	The LOG environment variable must be specified.

Options specific to *part-bench*:

*-e, --efficiency*
	Measure strong scaling by running the algorithm with different amounts of
	threads.

*-b, --baseline* <name>
*-s, --save-baseline* <name>
	Compare against a named baseline.  If *--save-baseline* is specified, the
	results of the benchmark will be saved and overwrite the previous ones.

# ENVIRONMENT VARIABLES

Users can use environment variables to alter the execution of these programs:

- _LOG=coupe_, for mesh-part, enable debug algorithm logging of coupe's
  algorithms,
- _RAYON_NUM_THREADS=n_ restricts the maximum number of threads to _n_.

# ALGORITHMS

The option *-a, --algorithm* defines an algorithm used to partition the input
mesh.  It can be specified multiple times to chain algorithms.

If a partition improving algorithm is in the begining of the chain, it will
be fed a default partition where all weights are in the same part.

For example,

	mesh-part --algorithm hilbert,4 --algorithm fm,0.05

Miscellaneous partitioning algorithms:

	*random*,PART_COUNT,[SEED=0]
		Creates a random partition

Number partitioning algorithms:++
These algorithms create partitions by only taking weights into account.

	*greedy*,PART_COUNT
		Greedy number partitioning algorithm

	*kk*,PART_COUNT
		Karmarkar-Karp, Least Difference Method

	*ckk*,TOLERANCE
		Complete Karmarkar-Karp

Number partition improving algorithms:++
These algorithms improve partitions by only taking weights into account.

	*vn-best*
		Steepest descent Vector of Numbers algorithm

	*vn-first*
		Descent Vector of Numbers algorithm

Geometric partitioning algorithms:++
These algorithms create partitions using cell coordinates.

	*rcb*,PART_COUNT,[TOLERANCE=0.05]
		Recursive Coordinate Biscection

	*hilbert*,PART_COUNT,[ORDER=12]
		Hilbert Curve

Geometric partition improving algorithms:++
These algorithms improve partitions using cell coordinates.

	*kmeans*
		Balanced k-means.

Graph partition improving algorithms:++
These algorithms improve partitions using the topology of the mesh.

	*fm*,[args...]
		Fidducia-Mattheyses algorithm.  Arguments (in this order):
		- MAX_IMBALANCE (default: initial imbalance)
		- MAX_BAD_MOVES_IN_A_ROW (default: 0)
		- MAX_PASSES (default: 0, i.e. UINTPTR_MAX)
		- MAX_MOVES_PER_PASS (default: 0, i.e. UINTPTR_MAX)

	*arcswap*,[MAX_IMBALANCE]
		Multi-threaded, FM-like algorithm.  By default this will not worsen the
		imbalance.

	*kl*,[MAX_BAD_MOVES_IN_A_ROW=1]
		Kernighan-Lin algorithm

METIS partitioning algorithms:++
These algorithms require *mesh-part* to be built with METIS support.

	*metis:recursive*,PART_COUNT
		METIS's recursive graph partitioning.

	*metis:kway*,PART_COUNT
		METIS's kway graph partitioning.

SCOTCH partitioning algorithms:++
These algorithms require *mesh-part* to be built with SCOTCH support.

	*scotch:std*,PART_COUNT
		The default partitioning algorithm from SCOTCH.

# PARTITION FILE

A partition file is a binary file that consists of a header followed by a list of
part IDs.  Below is the ABNF representation of the file:

```
file     = magic id-count ids
magic    = %x4d %x65 %x50 %x65  ; "MePe"
id-count = U64                  ; Number of weights
ids      = *U64                 ; id-count weights
```

where _U64_ is a little-endian 8-byte unsigned integer.

# SEE ALSO

*apply-part*(1) *part-info*(1) *weight-gen*(1)

# AUTHORS

This executable is part of coupe, which is maintained by Hubert Hirtz
<hubert@hirtz.pm> under the direction of Franck Ledoux <franck.ledoux@cea.fr>
and the supervision of Cédric Chevalier <cedric.chevalier@cea.fr> and Sébastien
Morais <sebastien.morais@cea.fr>.

For more information on coupe development, see
<https://github.com/LIHPC-Computational-Geometry/coupe>.
