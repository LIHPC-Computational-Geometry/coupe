mesh-refine(1)

# NAME

mesh-dup - Duplicate the elements of a mesh

# SYNOPSIS

*mesh-dup* [--times <n>] <input.mesh >output.mesh

# DESCRIPTION

mesh-dup copies the elements of a mesh a number of times so that the output mesh
forms a (hyper)grid of input meshs, mainly for performance tests on algorithms.

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

*mesh-refine*(1) *mesh-part*(1)

# AUTHORS

This executable is part of coupe, which is maintained by Hubert Hirtz
<hubert@hirtz.pm> under the direction of Franck Ledoux <franck.ledoux@cea.fr>
and the supervision of Cédric Chevalier <cedric.chevalier@cea.fr> and Sébastien
Morais <sebastien.morais@cea.fr>.

For more information on coupe development, see
<https://github.com/LIHPC-Computational-Geometry/coupe>.
