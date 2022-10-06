# [coupe]

A modular, multithreaded partitioning library.

Coupe implements a variety of algorithms that can be used to partition meshes,
graphs and numbers. See [the API docs][coupe] for a list. These algorithms can
be composed together to build relevant partitions of your data.

## Getting Coupe

### Released versions

The simplest way to use `Coupe` is from Rust, referencing `coupe` crate.

### Building from source

`Coupe` is written in Rust, so you'll need to grab a Rust installation in order to compile it. In general, `Coupe`
tracks the latest stable release of the Rust compiler.

To build the whole `Coupe` platform:

```shell
git clone https://github.com/LIHPC-Computational-Geometry/coupe/
cd coupe
cargo build --workspace --release
```

By default, `Coupe` tools need `Scotch` and `Metis`. It can be disabled using

```shell
cargo build --workspace --release --no-default-features
```

Else, if you want `Scotch` or `Metis` support, you might need to pass some information about header location.
On some systems, files like `scotch.h` are not in standard include directories. One can configure it
using `BINDGEN_EXTRA_CLANG_ARGS` environment variable.
It can be set from the shell or using `.cargo/config.toml` file like this (this example is for debian and ubuntu
systems)

```toml
[env]
BINDGEN_EXTRA_CLANG_ARGS = "-I/usr/include/scotch"
```

`Coupe` is relatively well-tested, including both unit tests and integration tests. To run the full test suite, use:

```shell
cargo test --all --workspace
```

from the repository root.

## Usage

### From the command-line

A list of tools is provided to work with coupe from the command-line, you may
find them, along with their documentation in the `tools/` directory.

### From Rust

See the API documentation on [docs.rs][coupe], and the `examples/` directory for
example usages of the library.

### From other languages

Coupe offers a C interface which can be found in the `ffi/` directory.

Bindings for other languages have not been made yet. If you end up developing
such bindings, please send us a note, so they can be shown here!

## Contributing

Contributions are welcome and accepted as pull requests on [GitHub][pulls].

You may also ask questions on the [discussion forum][discussions] and file bug
reports on the [issue tracker][issues].

## License

Licensed under either of

* Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

[coupe]: https://docs.rs/coupe

[discussions]: https://github.com/LIHPC-Computational-Geometry/coupe/discussions

[issues]: https://github.com/LIHPC-Computational-Geometry/coupe/issues

[pulls]: https://github.com/LIHPC-Computational-Geometry/coupe/pulls
