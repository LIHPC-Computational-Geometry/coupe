part-info(1)

# NAME

part-info - Print information about the quality a partition

# SYNOPSIS

*part-info* [options...]

# DESCRIPTION

part-info prints the following information:

- the imbalance for each criterion,
- the edge cut,
- the lambda-1 cut.

Only some specific mesh formats are supported.  See *apply-part*(1)'s *INPUT
FORMAT* for details.

# OPTIONS

*-h, --help*
	Show a help message and exit.

*--version*
	Show version information and exit.

*-E, --edge-weights* <variant>
	Change how edge weights are set.  Possible values are listed in
	*mesh-part*(1).

*-m, --mesh* <path>
	Required.  A path to the mesh that has been partitionned.

*-n, --parts* <path>
	The expected number of parts.  By default, part-info will take the actual
	number of parts (which can be lower).

*-p, --partition* <path>
	Required.  A path to the partition file.

*-w, --weights* <path>
	Required.  A path to the weight file used to partition the mesh.

# SEE ALSO

*mesh-part*(1) *apply-part*(1) *weight-gen*(1)

# AUTHORS

This executable is part of coupe, which is maintained by Hubert Hirtz
<hubert@hirtz.pm> under the direction of Franck Ledoux <franck.ledoux@cea.fr>
and the supervision of Cédric Chevalier <cedric.chevalier@cea.fr> and Sébastien
Morais <sebastien.morais@cea.fr>.

For more information on coupe development, see
<https://github.com/LIHPC-Computational-Geometry/coupe>.
