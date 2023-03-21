mesh-points(1)

# NAME

mesh-points - Extract the cell centers of the given mesh

# SYNOPSIS

*mesh-points* [input.mesh [output.dat]]

# DESCRIPTION

mesh-points extracts the cell centers of the given mesh.

If input.mesh is omitted or is -, it is read from standard input.
If output.dat is omitted or is -, it is written to standard output.

Only some specific mesh formats are supported.  See *apply-part*(1)'s *INPUT
FORMAT* for details.

# OPTIONS

*-h, --help*
	Show a help message and exit.

*--with-ids*
	Add a column with the cell IDs (refs).

# USAGE EXAMPLE WITH GNUPLOT

Using *--with-ids* on a 3D mesh, one ends up with a file with 4 columns. The
first three are the cell center coordinates, and the fourth is the cell ID:

```
1.2 4 6.6 0
3 12.5 9.01 4
etc.
```

Then the gnuplot commands can be used to generate a 3D, colored plot of those
points:

```
splot 'mesh.points' using 1:2:3:4 with points palette
```

# SEE ALSO

*apply-part*(1) *apply-weight*(1)

# AUTHORS

This executable is part of coupe, which is maintained by Hubert Hirtz
<hubert@hirtz.pm> under the direction of Franck Ledoux <franck.ledoux@cea.fr>
and the supervision of Cédric Chevalier <cedric.chevalier@cea.fr> and Sébastien
Morais <sebastien.morais@cea.fr>.

For more information on coupe development, see
<https://github.com/LIHPC-Computational-Geometry/coupe>.
