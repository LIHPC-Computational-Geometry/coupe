ARG BASEIMAGE=rust:latest

# BUILDER PATTERN

FROM $BASEIMAGE AS builder

WORKDIR /builder

COPY . .

RUN apt-get update
RUN apt-get -y install libclang-dev libmetis-dev libscotch-dev

ARG C_INCLUDE_PATH="/usr/include/x86_64-linux-musl"
ARG CPLUS_INCLUDE_PATH="/usr/include/x86_64-linux-musl"

RUN rustup target add x86_64-unknown-linux-musl
RUN cargo build --workspace --release --bins --target x86_64-unknown-linux-musl --no-default-features

# FINAL IMAGE

FROM alpine

WORKDIR /coupe

COPY --from=builder /builder/target/x86_64-unknown-linux-musl/release/apply-part /bin
COPY --from=builder /builder/target/x86_64-unknown-linux-musl/release/apply-weight /bin
COPY --from=builder /builder/target/x86_64-unknown-linux-musl/release/medit2svg /bin
COPY --from=builder /builder/target/x86_64-unknown-linux-musl/release/mesh-part /bin
COPY --from=builder /builder/target/x86_64-unknown-linux-musl/release/mesh-refine /bin
COPY --from=builder /builder/target/x86_64-unknown-linux-musl/release/mesh-reorder /bin
COPY --from=builder /builder/target/x86_64-unknown-linux-musl/release/part-bench /bin
COPY --from=builder /builder/target/x86_64-unknown-linux-musl/release/part-info /bin
COPY --from=builder /builder/target/x86_64-unknown-linux-musl/release/weight-gen /bin

COPY --from=builder /builder/examples/meshes meshes
