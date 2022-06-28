#include <stdio.h>
#include "../include/coupe.h"

#define POINT_COUNT 4
#define DIMENSION 2

int
main()
{
	uintptr_t partition[POINT_COUNT];

	double point_array[POINT_COUNT][DIMENSION] = {
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	};
	struct coupe_data *points = coupe_data_array(POINT_COUNT, COUPE_DOUBLE, point_array);

	int one = 1;
	struct coupe_data *weights = coupe_data_constant(POINT_COUNT, COUPE_INT, &one);

	uintptr_t iter_count = 1;
	double tolerance = 0.05;
	enum coupe_err err =
		coupe_rcb(partition, DIMENSION, points, weights, iter_count, tolerance);
	if (err != COUPE_ERR_OK) {
		fprintf(stderr, "Error: %s\n", coupe_strerror(err));
		coupe_data_free(points);
		coupe_data_free(weights);
		return 1;
	}

	printf("With 1 iteration (2 parts), RCB returned: %s\n", coupe_strerror(err));
	printf("partition:\n");
	printf("%ld %ld\n%ld %ld\n", partition[0], partition[1], partition[2], partition[3]);

	iter_count = 2;
	err = coupe_rcb(partition, DIMENSION, points, weights, iter_count, tolerance);
	if (err != COUPE_ERR_OK) {
		fprintf(stderr, "Error: %s\n", coupe_strerror(err));
		coupe_data_free(points);
		coupe_data_free(weights);
		return 1;
	}

	printf("With 2 iterations (4 parts), RCB returned: %s\n", coupe_strerror(err));
	printf("partition:\n");
	printf("%ld %ld\n%ld %ld\n", partition[0], partition[1], partition[2], partition[3]);

	coupe_data_free(points);
	coupe_data_free(weights);
	return 0;
}
