# libcoupemetis.so

This library is an implementation of [MeTiS] 5.1 using coupe's algorithms.

[MeTiS]: http://glaros.dtc.umn.edu/gkhome/metis/metis/overview

## Building

Build it with cargo.  It will produce a shared object and an archive for static
linking.

## Usage

This library is a drop-in replacement for `libmetis.so`.  Simply replace
`-lmetis` by `-lcoupemetis` in your linker flags:

```shell
# Change this
cc -o my_program -c main.c -lmetis

# To this
cc -o my_program -c main.c -lcoupemetis
```
