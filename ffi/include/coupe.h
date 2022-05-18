#ifndef COUPE_H
#define COUPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return type for coupe algorithms.
 *
 * You may use `coupe_strerror()` to print a user-friendly message relative to a
 * given error.
 */
enum coupe_err {
	/**
	 * Everything is fine, no error happened.
	 */
	COUPE_ERR_OK,
	/**
	 * Coupe failed to allocate something and aborted.
	 */
	COUPE_ERR_ALLOC,
	/**
	 * Coupe encountered a bug and crashed.
	 */
	COUPE_ERR_CRASH,
	/**
	 * Either the dimension is invalid (e.g. equals zero), or the algorithm
	 * does not support such dimension (e.g. Hilbert curves are only
	 * implemented in two dimensions for now).
	 */
	COUPE_ERR_BAD_DIMENSION,
	/**
	 * The algorithm does not support a given type.
	 */
	COUPE_ERR_BAD_TYPE,
	/**
	 * An bi-partitioning algorithm has been fed a partition with more than
	 * two parts.
	 */
	COUPE_ERR_BIPART_ONLY,
	/**
	 * Data sets passed to an algorithm don't have the same number of
	 * elements.
	 */
	COUPE_ERR_LEN_MISMATCH,
	/**
	 * No partition matching the given constraints have been found.
	 */
	COUPE_ERR_NOT_FOUND,
	/**
	 * Input contains negative values and such values are not supported.
	 */
	COUPE_ERR_NEG_VALUES,
};
/**
 * A descriptive message associated with the given error code.
 *
 * The result is a completely static, nul-terminated string and doesn't need to
 * be deallocated.
 */
const char *coupe_strerror(enum coupe_err err);

/**
 * A data set, used to feed values to algorithms.
 *
 * Several types of data sets can be constructed, depending on your needs:
 *
 * - `coupe_data_array()` uses a plain contiguous array of values,
 * - `coupe_data_constant()` repeats the same value a certain amount of times,
 * - `coupe_data_fn()` calls a function for each value as needed.
 */
typedef struct coupe_data coupe_data;
/**
 * Data types.
 *
 * Some algorithms can be fed vertex weights and edge weights of different
 * types.  You'll have to see what they support individually.
 *
 * Data sets' elements are expected to match the type given with this enum and,
 * depending on the context, sometimes several values have to be fed for each
 * item.  More info on individual algorithms.
 */
enum coupe_type {
	/** `int` */
	COUPE_INT,
	/** `int64_t` */
	COUPE_INT64,
	/** `double` */
	COUPE_DOUBLE,
};
/**
 * Free the memory used by the given data set.
 *
 * This function:
 *
 * - can be called with a NULL pointer,
 * - can only be called once on the same data set,
 * - does not free any memory the data set points to (e.g. pointers passed to
 *   `coupe_data_array` and `coupe_data_constant`),
 * - cannot be called while an algorithm is running on the data set.
 */
void coupe_data_free(coupe_data *data);
/**
 * A data set that uses values from a contiguous array.
 *
 * This function returns a data set, which is used to feed numbers and points
 * to coupe's algorithms.  These values must be ordered and their order must
 * match those of the given `partition` arrays (see `coupe_hilbert()` for
 * example).
 *
 * The data set returned by this function retrieves values from the contiguous
 * array `data`, which must uphold the following constraints:
 *
 * - `data` must be at least of length `len`,
 * - `data` must be available during the lifetime of the data set, and
 * - it must be possible to access `data` from several threads at the same time.
 *
 * This data set can be used several times and so concurrently.  When unused, it
 * can be freed with `coupe_data_free()`.
 *
 * This function returns NULL on allocation failure.
 */
coupe_data *coupe_data_array(uintptr_t len, enum coupe_type type, const void *data);
/**
 * A data set that copies the same value a given amount of times.
 *
 * This function returns a data set, which is used to feed numbers and points
 * to coupe's algorithms.  These values must be ordered and their order must
 * match those of the given `partition` arrays (see `coupe_greedy()` for
 * example).
 *
 * The data set returned by this function copies the value beind the given
 * pointer, which must uphold the following constraints:
 *
 * - `value` must be available during the lifetime of the data set, and
 * - it must be possible to access `value` from several threads at the same time.
 *
 * This data set can be used several times and so concurrently.  When unused, it
 * can be freed with `coupe_data_free()`.
 *
 * This function returns NULL on allocation failure.
 */
coupe_data *coupe_data_constant(uintptr_t len, enum coupe_type type, const void *value);
/**
 * A data set that calls a function to retrieve values.
 *
 * This function returns a data set, which is used to feed numbers and points
 * to coupe's algorithms.  These values must be ordered and their order must
 * match those of the given `partition` arrays (see `coupe_rcb()` for
 * example).
 *
 * The data set returned by this function retrieves values by calling the given
 * callback `i_th` as needed.  This callback accepts two arguments:
 *
 * - `const void *context`, which is set to the same value as the `context`
 *   given in `coupe_data_fn`,
 * - `uintptr_t index`, which is the index of the element to be retrieved.
 *
 * The callback `i_th` must assume the following conditions:
 *
 * - it must not assume the order in which it is called. For example, an
 *   algorithm might do this:
 *
 *   ```
 *	   i_th(context, 0);
 *	   i_th(context, 4572);
 *	   i_th(context, 19);
 *   ```
 *
 * - `i_th(context, index)` must be valid for all values of `index` between 0
 *   (inclusive) and `len` (exclusive),
 * - some values might not be retrieved,
 * - some values might be retrieved more than once,
 * - `i_th` might be called from several threads at a time, this means access to
 *   `context` must be thread-safe, and is why it is sealed by a const pointer.
 *
 * This data set can be used several times and so concurrently.  When unused, it
 * can be freed with `coupe_data_free()`.
 *
 * This function returns NULL on allocation failure.
 */
coupe_data *coupe_data_fn(const void *context, uintptr_t len, enum coupe_type type,
		const void *(*i_th)(const void *, uintptr_t));

/**
 * Adjacency structure for use with topologic algorithms.
 *
 * This type can be constructed in several ways:
 *
 * - `coupe_adjncy_csr()` uses an adjacency matrix in the CSR format, in the
 *   same way as METIS,
 * - `coupe_adjncy_csr_unchecked()` is the same, except the matrix structure is
 *   not checked at runtime.
 */
typedef struct coupe_adjncy coupe_adjncy;
/**
 * Free the memory used by the given adjacency structure.
 *
 * This function:
 *
 * - can be called with a NULL pointer,
 * - can only be called once on the same structure,
 * - does not free any memory it points to (e.g. the actual CSR arrays),
 * - cannot be called while an algorithm is using it.
 */
void coupe_adjncy_free(coupe_adjncy *adjncy);
/**
 * An adjacency matrix in the CSR format.
 *
 * The following properties are expected:
 *
 * - `xadj` is a contiguous array of `size + 1` elements,
 * - `adjncy` is a contiguous array of N elements, where N is the last element
 *   of `xadj`,
 * - `data` is a contiguous array of N elements, where N is the last element
 *   of `xadj`,
 * - all given pointers are accessible concurrently from multiple threads.
 *
 * This function does the following checks on the adjacency matrix structure and
 * returns NULL on failure:
 *
 * - `xadj` must be sorted,
 * - elements of `xadj` must not exceed `UINTPTR_MAX / 2`,
 * - `adjncy[xadj[i]..xadj[i+1]]` must be sorted for each `i`,
 * - elements of `adjncy` must be between 0 (inclusive) and `size` (exclusive).
 *
 * This adjacency structure can be used several times and so concurrently.  When
 * unused, it can be freed with `coupe_adjncy_free()`.
 *
 * This function returns NULL on allocation failure.
 */
coupe_adjncy *coupe_adjncy_csr(uintptr_t size,
		const uintptr_t *xadj, const uintptr_t *adjncy,
		enum coupe_type data_type, const void *data);
/**
 * An adjacency matrix in the CSR format (unchecked constructor).
 *
 * See the documentation of `coupe_adjncy_csr()` for details.
 *
 * This function does not check the adjacency matrix structure.  It is up to the
 * user to ensure all invariants are met.
 */
coupe_adjncy *coupe_adjncy_csr_unchecked(uintptr_t size,
		const uintptr_t *xadj, const uintptr_t *adjncy,
		enum coupe_type data_type, const void *data);

/************************/
/* Geometric algorithms */
/************************/

/**
 * Algorithm: Recursive Coordinate Biscection.
 *
 * See <https://docs.rs/coupe/latest/coupe/struct.Rcb.html> for a description of
 * the algorithm. The meaning of the parameters are explained below the example.
 *
 * This function returns an error for cases covered by #coupe_err:
 * - `dimension` must be 2 or 3,
 * - `points` and `weights` must hold the same number of elements.
 *
 * This function also assumes the following invariant is met:
 * - `partition` points to a contiguous array of N `uintptr_t` elements, where N
 *   is the size of either data set.
 *
 * The result is stored in `partition`.
 *
 * @note
 * This function might return #COUPE_ERR_NOT_FOUND.  It means one of the splits
 * did not meet the input tolerance.  It is possible, however, that the overall
 * imbalance still is within bounds.  It is also possible of the contrary: the
 * overall imbalance might be out of bounds without RCB returning an error.
 */
enum coupe_err coupe_rcb(uintptr_t *partition, uintptr_t dimension,
		const coupe_data *points, const coupe_data *weights,
		uintptr_t iter_count, double tolerance);
/**
 * Algorithm: Recursive Inertial Biscection.
 *
 * See <https://docs.rs/coupe/latest/coupe/struct.Rib.html> for a description of
 * the algorithm. The meaning of the parameters are explained below the example.
 *
 * See the documentation of `coupe_rcb()` for details.
 *
 * @note
 * This algorithm does not fully support data sets yet.  If either `points` or
 * `weights` are not made from the `coupe_data_array()`, they are copied into a
 * newly allocated array.
 */
enum coupe_err coupe_rib(uintptr_t *partition, uintptr_t dimension,
		const coupe_data *points, const coupe_data *weights,
		uintptr_t iter_count, double tolerance);
/**
 * Algorithm: Hilbert space-filling curve
 *
 * See <https://docs.rs/coupe/latest/coupe/struct.HilbertCurve.html> for a
 * description of the algorithm. The meaning of the parameters are explained
 * below the example.
 *
 * This function returns an error for cases covered by #coupe_err:
 * - `points` and `weights` must hold the same number of elements,
 * - `order` must be below 64.
 *
 * This function also assumes the following invariants are met:
 * - `points` are in 2D,
 * - `weights` are of type `double`,
 * - `partition` points to a contiguous array of N `uintptr_t` elements, where N
 *   is the size of either data set.
 *
 * The result is stored in `partition`.
 *
 * @note
 * This algorithm does not fully support data sets yet.  If either `points` or
 * `weights` are not made from the `coupe_data_array()`, they are copied into a
 * newly allocated array.
 */
enum coupe_err coupe_hilbert(uintptr_t *partition,
		const coupe_data *points,
		const coupe_data *weights,
		uintptr_t part_count, uint32_t order);

/**********************************/
/* Number-partitioning algorithms */
/**********************************/

/**
 * Algorithm: Greedy number partitioning algorithm
 *
 * See <https://docs.rs/coupe/latest/coupe/struct.Greedy.html> for a description
 * of the algorithm. The meaning of the parameters are explained below the
 * example.
 *
 * This function also assumes the following invariants are met:
 * - floating point `weights` are not NaN,
 * - `partition` points to a contiguous array of N `uintptr_t` elements, where N
 *   is the size of the weight data set.
 *
 * The result is stored in `partition`.
 */
enum coupe_err coupe_greedy(uintptr_t *partition, const coupe_data *weights,
		uintptr_t part_count);
/**
 * Algorithm: Karmarkar-Karp
 *
 * See <https://docs.rs/coupe/latest/coupe/struct.KarmarkarKarp.html> for a
 * description of the algorithm. The meaning of the parameters are explained
 * below the example.
 *
 * This function returns an error for cases covered by `enum coupe_err`:
 * - `weight_type` can be either type.
 *
 * This function also assumes the following invariants are met:
 * - floating point `weights` are not NaN,
 * - `partition` points to a contiguous array of N `uintptr_t` elements, where N
 *   is the size of the weight data set.
 *
 * The result is stored in `partition`.
 */
enum coupe_err coupe_karmarkar_karp(uintptr_t *partition, const coupe_data *weights,
		uintptr_t part_count);
/**
 * Algorithm: Complete Karmarkar-Karp
 *
 * See <https://docs.rs/coupe/latest/coupe/struct.CompleteKarmarkarKarp.html>
 * for a description of the algorithm. The meaning of the parameters are
 * explained below the example.
 *
 * This function also assumes the following invariants are met:
 * - floating point `weights` are not NaN,
 * - `partition` points to a contiguous array of N `uintptr_t` elements, where N
 *   is the size of the weight data set.
 *
 * The result is stored in `partition`.
 */
enum coupe_err coupe_karkarkar_karp_complete(uintptr_t *partition, const coupe_data *weights,
		double tolerance);

/************************/
/* Topologic algorithms */
/************************/

/**
 * Algorithm: Fiduccia-Mattheyses
 *
 * See <https://docs.rs/coupe/latest/coupe/struct.FiducciaMattheyses.html> for a
 * description of the algorithm. The meaning of the parameters are explained
 * below the example.
 *
 * This function returns an error for cases covered by #coupe_err:
 * - there cannot be more than two parts,
 * - `adjncy` must be of type #COUPE_INT64.
 *
 * This function also assumes the following invariants are met:
 * - floating point `weights` are not NaN,
 * - `partition` points to a contiguous array of N `uintptr_t` elements, where N
 *   is the size of either the weight data set or the adjacency structure,
 * - the elements of `partition` must start from zero and must be continuous:
 *   the number of parts is taken from the maximum of this array, plus one.
 *
 * The result is stored in `partition`.
 *
 * @note
 * - zero values for `max_passes` and `max_moves_per_pass` are interpreted as
 *   abscence of value (no limit on the number of passes or moves per pass),
 *   while zero values for `max_bad_moves_in_a_row` prevents the use of bad
 *   moves,
 * - if `max_imbalance` is negative, it will be set to the imbalance of the
 *   input partition.
 */
enum coupe_err coupe_fiduccia_mattheyses(uintptr_t *partition,
		const coupe_adjncy *adjncy, const coupe_data *weights,
		uintptr_t max_passes, uintptr_t max_moves_per_pass,
		double max_imbalance, uintptr_t max_bad_moves_in_a_row);

#ifdef __cplusplus
}
#endif

#endif
