mesh-refine(1)

# NAME

mesh-refine - Refine a mesh

# SYNOPSIS

*mesh-refine* [--times <n>] <input.mesh >output.mesh

# DESCRIPTION

mesh-refine splits the cell of a mesh from its standard input into smaller
cells and prints the resulting mesh on its standard output, mainly for
performance tests on algorithms.

mesh-refine only works on 2D meshes composed exclusively of triangles and
quadrilaterals.  It splits either of them into four equal parts at each
iteration.

Only MEDIT meshes are supported, for now.  See *apply-part(1)*'s *INPUT FORMAT*
for details.

# OPTIONS

*-h, --help*
	Show a help message and exit.

*-f, --format* <format>
	Set the output format.  See *apply-part(1)*'s *OUTPUT FORMAT* for the list
	of accepted values.

*-n, --times* <n>
	Iterate a given number of times.  By default, only once.

# SEE ALSO

*mesh-dup*(1) *mesh-part*(1)

# AUTHORS

This executable is part of coupe, which is maintained by Hubert Hirtz
<hubert@hirtz.pm> under the direction of Franck Ledoux <franck.ledoux@cea.fr>
and the supervision of Cédric Chevalier <cedric.chevalier@cea.fr> and Sébastien
Morais <sebastien.morais@cea.fr>.

For more information on coupe development, see
<https://github.com/LIHPC-Computational-Geometry/coupe>.
