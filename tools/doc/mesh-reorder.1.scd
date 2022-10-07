mesh-reorder(1)

# NAME

mesh-reorder - Change the order of the vertices and the elements of a mesh

# SYNOPSIS

*mesh-reorder* [input.mesh [output.mesh]]

# DESCRIPTION

mesh-reorder shuffles the vertices and the elements of a mesh, without altering
its mathematical structure: only the order in which they appear in the file is
changed.

If input.mesh is omitted or is -, it is read from standard input.
If output.mesh is omitted or is -, it is written to standard output.

The tool can be used in conjunction of *mesh-reorder* or *mesh-dup*, which both
produce very typical element orderings and may disrupt algorithms.

Only some specific mesh formats are supported.  See *apply-part*(1)'s *INPUT
FORMAT* for details.

# OPTIONS

*-h, --help*
	Show a help message and exit.

*-f, --format* <format>
	Override the output format.  By default, the file format is inferred from
	the file extension.  See *apply-part*(1)'s *OUTPUT FORMAT* for more info.

# SEE ALSO

*mesh-dup*(1) *mesh-refine*(1)

# AUTHORS

This executable is part of coupe, which is maintained by Hubert Hirtz
<hubert@hirtz.pm> under the direction of Franck Ledoux <franck.ledoux@cea.fr>
and the supervision of Cédric Chevalier <cedric.chevalier@cea.fr> and Sébastien
Morais <sebastien.morais@cea.fr>.

For more information on coupe development, see
<https://github.com/LIHPC-Computational-Geometry/coupe>.
