#!/bin/sh
set -e

# Cargo target directory
TARGET="${TARGET:-target}"

# Generate coverage for coupe and tools
# We need to call this before coverage for ffi, because cargo-llvm-cov removes
# all previous profraw files before runs.
# No default features to avoid linking to SCOTCH and METIS.
cargo +nightly llvm-cov \
	--doctests \
	--no-report \
	--no-default-features \
	--workspace

# Building on nightly since we used nightly for cargo-llvm-cov.
RUSTFLAGS="-Cinstrument-coverage $RUSTFLAGS" cargo +nightly build -p coupe-ffi

mkdir -p "$TARGET/ffi-examples"
for f in ffi/examples/*.c
do
	example="$(basename "$f" .c)"

	clang "$f" \
		-o "$TARGET/ffi-examples/$example" \
		-L"$TARGET/debug" \
		-lcoupe \
		-g -Wall -Wextra -Werror

	LLVM_PROFILE_FILE="$TARGET/llvm-cov-target/coupe-$example.profraw" \
		LD_LIBRARY_PATH="$TARGET/debug" \
		"$TARGET/ffi-examples/$example"
done

# Feed ffi objects to cargo-llvm-cov
mv "$TARGET/debug/libcoupe.so" "$TARGET/llvm-cov-target/debug/"
mv "$TARGET/debug/libcoupe.a" "$TARGET/llvm-cov-target/debug/"

cargo +nightly llvm-cov \
	report \
	--lcov \
	--output-path lcov.info
