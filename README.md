# [coupe]

A modular, multi-threaded partitioning library.

Coupe implements a variety of algorithms that can be used to partition meshes,
graphs and numbers.  See [the API docs][coupe] for a list.  These algorithms can
be composed together to build relevant partitions of your data.

## Usage

### From the command-line

A list of tools is provided to work with coupe from the command-line, you may
find them, along with their documentation in the `tools/` directory.

### From Rust

See the API documentation on [docs.rs][coupe], and the `examples/` directory for
example usages of the library.

### From other languages

Coupe can only be used from Rust, for now.  Interoperability with other
languages will be available through C bindings, and is a work in progress.  See
[the relevant PR][67] for more information.

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


[67]: https://github.com/LIHPC-Computational-Geometry/coupe/pull/67
[coupe]: https://docs.rs/coupe
[discussions]: https://github.com/LIHPC-Computational-Geometry/coupe/discussions
[issues]: https://github.com/LIHPC-Computational-Geometry/coupe/issues
[pulls]: https://github.com/LIHPC-Computational-Geometry/coupe/pulls
