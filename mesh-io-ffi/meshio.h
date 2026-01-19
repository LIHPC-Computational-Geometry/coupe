#ifndef LIBMESHIO_H
#define LIBMESHIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ERROR_GENERIC             -1
#define ERROR_BAD_HEADER          -2
#define ERROR_UNSUPPORTED_VERSION -3

/** Read a partition file from a file descriptor.
 *
 * Returns a negative number on error.
 * The partition must be freed using partition_free.
 * The file descriptor is closed. */
int mio_partition_read(uint64_t *size, uint64_t **partition, int fd);
/** Write a partition file to a file descriptor.
 *
 * Returns a negative number on error.
 * The file descriptor is left open. */
int mio_partition_write(int fd, uint64_t size, const uint64_t *partition);
/** Free a partition.
 *
 * Both arguments must match the result of mio_partition_read. */
void mio_partition_free(uint64_t size, uint64_t *partition);

/** An array of weights, which can either be floats or integers. */
struct mio_weights;
/** Read weights from the given file descriptor.
 *
 * Returns NULL on error.
 * The result must be freed using mio_weights_free.
 * The file descriptor is closed. */
struct mio_weights *mio_weights_read(int fd);
/** The number of weights in the given weight array. */
uint64_t mio_weights_count(struct mio_weights *weights);
/** Copy and convert the first criterion of the array into the prealocated
 * `criterion` array.
 *
 * The given pointer must be allocated with enough memory to contain all weights
 * (must point to at least mio_weights_count(weights) doubles).
 * Integer weights are converted to double. */
void mio_weights_first_criterion(double *criterion, struct mio_weights *weights);
/** Free a weight array. */
void mio_weights_free(struct mio_weights *weights);

/** A mesh. */
struct mio_mesh;
/** Read a mesh from the given file descriptor.
 *
 * Returns NULL on error.
 * The return value must be freed with mio_mesh_free.
 * The file descriptor is closed. */
struct mio_mesh *mio_mesh_read(int fd);
/** Free the given mesh. */
void mio_mesh_free(struct mio_mesh *mesh);
/** The dimension of the mesh. */
int mio_mesh_dimension(struct mio_mesh *mesh);
/** The number of nodes/vertices in the mesh. */
uint64_t mio_mesh_node_count(struct mio_mesh *mesh);
/** The coordinates of the given node/vertex. */
const double *mio_mesh_coordinates(struct mio_mesh *mesh, uintptr_t node_idx);

/** The number of elements/cells in the mesh. */
uint64_t mio_mesh_element_count(struct mio_mesh *mesh);
/** Information about an element. */
struct mio_element {
    int dimension;
    int node_count;
    /** node_count indices that can be passed to mio_mesh_coordinates. */
    const uintptr_t *nodes;
};
/** Retrieve information about an element. */
void mio_mesh_element(struct mio_element *element, struct mio_mesh *mesh, uintptr_t element_idx);

#ifdef __cplusplus
}
#endif

#endif
