mesh-svg(1)

# NAME

mesh-svg - Render a 2D mesh into a SVG image

# SYNOPSIS

*mesh-svg* [input.mesh [output.svg]]

# DESCRIPTION

mesh-svg renders a 2D mesh as a SVG, optimizing for output size.

If input.mesh is omitted or is -, it is read from standard input.
If output.svg is omitted or is -, it is written to standard output.

Only some specific mesh formats are supported.  See *apply-part*(1)'s *INPUT
FORMAT* for details.

# OPTIONS

*-h, --help*
	Show a help message and exit.

*--version*
	Show version information and exit.

# SEE ALSO

*apply-part*(1) *apply-weight*(1)

# AUTHORS

This executable is part of coupe, which is maintained by Hubert Hirtz
<hubert@hirtz.pm> under the direction of Franck Ledoux <franck.ledoux@cea.fr>
and the supervision of Cédric Chevalier <cedric.chevalier@cea.fr> and Sébastien
Morais <sebastien.morais@cea.fr>.

For more information on coupe development, see
<https://github.com/LIHPC-Computational-Geometry/coupe>.
