apply-part(1)

# NAME

apply-part - Colorize a mesh with a partition file for visualization

apply-weight - Colorize a mesh with a weight file for visualization

# SYNOPSIS

*apply-part* --mesh <path> --partition <path> >output.mesh

*apply-weight* --mesh <path> --weights <path> >output.mesh

# DESCRIPTION

apply-part (apply-weight) merges a partition file (resp. weight file) into
the free-form fields of a MEDIT mesh file, so that the partition (resp. weight
distribution) can be visualized with medit tools.

The output mesh is written on standard output.

Only MEDIT meshes are supported, for now.  See *INPUT FORMAT* for details.

# OPTIONS

*-h, --help*
	Show a help message and exit.

*-f, --format* <format>
	Set the output format.  See *OUTPUT FORMAT* for the list of accepted
	values.

*-m, --mesh* <path>
	Use the given mesh file as template.

*-p --partition* <path>
	Use the given partition file.  This file is expected to come from
	*mesh-part*(1) with the same mesh.

*-w --weights* <path>
	Use the given weight file.  This file is expected to come from
	*weight-gen*(1) with the same mesh.

# INPUT FORMAT

These programs can accept multiple mesh formats, and will detect automatically
the correct version to use, regardless of the file extension.  There is
currently no way to force a particular format for input.

The supported mesh formats are:

- MEDIT, both ASCII and binary variants,
- VTK, legacy ASCII and binary variants.  The binary variant assumes little
  endian numbers.

# OUTPUT FORMAT

The *-f, --format* option changes the output format of the program.  These
values are currently accepted:

- _meshb_ (default): output MEDIT binary,
- _mesh_: output MEDIT ASCII.

In general, you'll want the default _meshb_ format.  It is faster to read from and
write to.  Use the ASCII format for debugging or for compatibility.

# SEE ALSO

*mesh-part*(1) *part-info*(1) *weight-gen*(1)

# AUTHORS

This executable is part of coupe, which is maintained by Hubert Hirtz
<hubert@hirtz.pm> under the direction of Franck Ledoux <franck.ledoux@cea.fr>
and the supervision of Cédric Chevalier <cedric.chevalier@cea.fr> and Sébastien
Morais <sebastien.morais@cea.fr>.

For more information on coupe development, see
<https://github.com/LIHPC-Computational-Geometry/coupe>.
