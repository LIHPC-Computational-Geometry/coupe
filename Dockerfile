# BUILDER PATTERN

FROM rust:1 AS builder

WORKDIR /builder

COPY . .

RUN apt-get update -y && apt-get install -y --no-install-recommends libclang-dev libmetis-dev libscotch-dev

ARG BINDGEN_EXTRA_CLANG_ARGS="-I/usr/include/scotch"

RUN cargo install --path tools --root /builder/install --features metis,scotch,intel-perf

# FINAL IMAGE

FROM debian:stable-slim

WORKDIR /coupe

COPY --from=builder /usr/lib/x86_64-linux-gnu/libscotch* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libmetis* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /builder/install/bin /usr/bin
