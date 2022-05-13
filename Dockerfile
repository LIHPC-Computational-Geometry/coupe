ARG BASEIMAGE=rust:slim-bullseye

# BUILDER PATTERN

FROM $BASEIMAGE AS builder

WORKDIR /builder

COPY . .

RUN apt-get update -y && apt-get install -y libclang-dev libmetis-dev libscotch-dev

ARG BINDGEN_EXTRA_CLANG_ARGS="-I/usr/include/scotch"

RUN cargo install --path tools --root /builder/install

# FINAL IMAGE

FROM debian:bullseye-slim

WORKDIR /coupe

COPY --from=builder /builder/install/bin /bin

COPY --from=builder /builder/examples/meshes meshes
