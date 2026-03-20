# libcoupe.so

C interface to coupe.

## Building

You can build this project with cargo, or use the provided Makefile:

```
make
sudo make install
```

Both a static (`libcoupe.a`) and a dynamic (`libcoupe.so`) library are produced.

**Note:** while we try our best to limit panics, coupe is not free of them. We
recommend packagers to keep the default `panic = "unwind"` or
`RUSTFLAGS="-Cpanic=unwind"` flag, so that they can be caught and do not abort
the whole process.

## Usage

These bindings target C99, but should work with later versions of the language.

A couple example usages can be found in the `examples/` directory.

## Documentation

The full documentation is written as doc comments in `include/coupe.h` and
should be seamlessly understood by IDEs and language servers.

A simple `Doxyfile` and make target are also provided to generate HTML and LaTeX
documents:

```
make doc
```

Generated documentation is placed inside a newly created `build/` directory.
