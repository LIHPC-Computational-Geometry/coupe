#include <stdio.h>
#include "../include/coupe.h"

#define POINT_COUNT 9
#define DIMENSION 2

static void
print_partition(uintptr_t partition[POINT_COUNT])
{
	printf("\n"
	       "  %ld---%ld---%ld\n"
	       "  |   |   |\n"
	       "  %ld---%ld---%ld\n"
	       "  |   |   |\n"
	       "  %ld---%ld---%ld\n"
	       "\n",
	       partition[2], partition[5], partition[8],
	       partition[1], partition[4], partition[7],
	       partition[0], partition[3], partition[6]);
}

int
main()
{
	/* Let's define a graph:
	 *
	 *     Node IDs:       Weights:
	 *
	 *     2    5    8     3    4    5
	 *      +---+---+       +---+---+
	 *      |   |   |       |   |   |
	 *     1+---4---+7     2+---3---+4
	 *      |   |   |       |   |   |
	 *      +---+---+       +---+---+
	 *     0    3    6     1    2    3
	 */
	double point_array[POINT_COUNT][DIMENSION] = {
		{0.0, 0.0},
		{0.0, 1.0},
		{0.0, 2.0},
		{1.0, 0.0},
		{1.0, 1.0},
		{1.0, 2.0},
		{2.0, 0.0},
		{2.0, 1.0},
		{2.0, 2.0},
	};

	int weight_array[POINT_COUNT] = { 1, 2, 3, 2, 3, 4, 3, 4, 5 };
	coupe_data *weights = coupe_data_array(POINT_COUNT, COUPE_INT, weight_array);
	if (weights == NULL) {
		fprintf(stderr, "Out of memory\n");
		return 1;
	}

	uintptr_t xadj[POINT_COUNT+1] = {0, 2, 5, 7, 10, 14, 17, 19, 22, 24};
	uintptr_t adjncy[24] =
	{1, 3, 0, 2, 4, 1, 5, 0, 4, 6, 1, 3, 5, 7, 2, 4, 8, 3, 7, 4, 6, 8, 5, 7};
	int64_t edge_weights[24] =
	{1, 1, 1, 2, 2, 2, 3, 1, 2, 2, 2, 2, 3, 3, 3, 2, 4, 2, 3, 3, 3, 4, 4, 4};
	struct coupe_adjncy *adjacency =
		coupe_adjncy_csr(POINT_COUNT, xadj, adjncy, COUPE_INT64, edge_weights);
	if (adjacency == NULL) {
		fprintf(stderr, "Either out of memory, or invalid adjacency structure\n");
		coupe_data_free(weights);
		return 1;
	}

	uintptr_t partition[POINT_COUNT];

	/* Let's run the Karmarkar-Karp algorithm on this graph: */
	uintptr_t part_count = 2;
	enum coupe_err err = coupe_karmarkar_karp(partition, weights, part_count);
	if (err != COUPE_ERR_OK) {
		fprintf(stderr, "Error: %s\n", coupe_strerror(err));
		coupe_adjncy_free(adjacency);
		coupe_data_free(weights);
		return 1;
	}

	printf("Initial partitioning with Karmarkar-Karp:\n");
	print_partition(partition);

	/* Let's refine the partition with Fiduccia-Mattheyses: */
	err = coupe_fiduccia_mattheyses(partition, adjacency, weights, 0, 0, 0.3, 0);
	if (err != COUPE_ERR_OK) {
		fprintf(stderr, "Error: %s\n", coupe_strerror(err));
		coupe_adjncy_free(adjacency);
		coupe_data_free(weights);
		return 1;
	}

	printf("Partition refined by Fiduccia-Mattheyses:\n");
	print_partition(partition);

	coupe_adjncy_free(adjacency);
	coupe_data_free(weights);
	return 0;
}
