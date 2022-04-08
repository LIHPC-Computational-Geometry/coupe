weight-gen(1)

# NAME

weight-gen - Generate a distribution of weights for a given mesh

# SYNOPSIS

*weight-gen* [options...] <input.mesh >output.weights

# DESCRIPTION

weight-gen generates weights for a given mesh provided to standard input, and
writes the output weight distribution in standard output.

Unless *-i, --integers* is specified, the generated weights are floating-point
numbers.

Only MEDIT ASCII meshes are supported, for now.

See *WEIGHT FILE* for a description of the output format.

# OPTIONS

*-h, --help*
	Show a help message and exit.

*-d, --distribution* <spec>
	Required.  The definition of a weight distribution.  This option must be
	specified once per criterion.  See *DISTRIBUTION FORMAT* and *SUPPORTED
	DISTRIBUTIONS* for more information.

*-i, --integers*
	Truncate weights and generate an integer weight distribution instead.

# DISTRIBUTION FORMAT

The option *-d, --distribution* defines how input weights are laid out.  It
can be specified multiple times, one for each criterion.  Its value must
follow the following syntax:

```
distribution  :=  name *( "," param)
param         :=  value / axis
name          :=  STRING
value         :=  FLOAT
axis          :=  "x" / "y" / "z" / "0" / "1" / "2"
```

For example, the following will lay out weights in ascending order on the
horizontal (X, abscissa) axis, from 0 to 100:

	weight-gen --distribution linear,x,0,100

And the following example shows how to set the weights to the same value:

	weight-gen --distribution constant,4.2

# SUPPORTED DISTRIBUTIONS

The following distributions are supported:

*constant*,VALUE
	Set all weights to be equal to the given VALUE.

*linear*,AXIS,FROM,TO
	Weights follow a linear slope on the given axis, where the points at the
	lowest coordinate on that axis are assigned the value FROM, and the one
	at the highest coordinate are assigned the value TO.

# WEIGHT FILE

A weight file is a binary file that consists of a header followed by a list of
weights, which can either be integers or floating points.  Below is the ABNF
representation of the file:

```
file            = magic version flags criterion-count weight-count weights
magic           = %x4d %x65 %x57 %x65  ; "MeWe"
version         = %x01                 ; Version 1
flags           = %x00-FF              ; see the Flags paragraph
criterion-count = U16                  ; Number of criteria
weight-count    = U64                  ; Number of weights per criterion
weights         = *I64 / *F64          ; criterion-count times weight-count weights
```

Some notes:

- Weights are laid out as weight-count arrays of criterion-count items.
- _U16_ is a little-endian 2-byte unsigned integer.
- _U64_ is a little-endian 8-byte unsigned integer.
- _I64_ is a little-endian 8-byte signed integer.
- _F64_ is a little-endian-encoded binary64 IEEE 754-2008 floating point.

The _flags_ byte has the following meaning, from the least significant bit to
the most:

- The first bit is 1 if weights are integers (_I64_), or 0 if weights are floats
  (_F64_),
- The other bits are left unspecified.

# SEE ALSO

*mesh-part*(1)

# AUTHORS

This executable is part of coupe, which is maintained by Hubert Hirtz
<hubert@hirtz.pm> under the direction of Franck Ledoux <franck.ledoux@cea.fr>
and the supervision of Cédric Chevalier <cedric.chevalier@cea.fr> and Sébastien
Morais <sebastien.morais@cea.fr>.

For more information on coupe development, see
<https://github.com/LIHPC-Computational-Geometry/coupe>.
