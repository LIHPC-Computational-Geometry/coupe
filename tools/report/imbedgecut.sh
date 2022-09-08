#!/bin/sh
set -e

. "$(dirname "$0")"/common.sh
PATH="$TARGET_DIR/release:$PATH"

ARGS="-m ../meshes/triangles_small.meshb -w /dev/shm/triangles.weights --edge-weights linear"
ALGORITHMS="
-a random,2
-a random,2 -a arcswap,0.05
-a random,2 -a arcswap,0.05 -a arcswap,0.05
-a rcb,1
-a rcb,1 -a arcswap,0.05
-a rcb,1 -a vn-best
-a rcb,1 -a vn-best -a arcswap,0.05
-a hilbert,2
-a hilbert,2 -a arcswap,0.05
-a hilbert,2 -a vn-best
-a hilbert,2 -a vn-best -a arcswap,0.05
-a kk,2
-a kk,2 -a arcswap,0.05
-a kk,2 -a vn-best -a arcswap,0.05
"

tmpdir=$(mktemp -d /dev/shm/imbedgecut.XXXXXX)
trap "rm -rf $tmpdir" 0 2 3 15

echo "$ALGORITHMS" | while read -r algorithm
do
	if [ -z "$algorithm" ]
	then continue
	fi

	iter_count=256
	if test "${algorithm#*arcswap}" = "$algorithm"
	then iter_count=1
	fi

	say Running "$algorithm ($iter_count times)" >&2

	for _ in $(seq $iter_count)
	do
		info=$(
			RAYON_NUM_THREADS=4 mesh-part $ARGS $algorithm |
				part-info $ARGS -p /dev/stdin
		)
		edge_cut=$(echo "$info" | grep edge | cut -f2 -d:)
		imbalance=$(echo "$info" | grep imbalances | sed 's/.*\[\(.*\)\]/\1/')
		echo "$edge_cut;$imbalance"
	done >"$tmpdir/$algorithm"
done

say Drawing histo.svg
gnuplot <<EOF
set term svg size 1152,648
set output 'histo.svg'
set datafile separator ';'
set xlabel 'edge cut (log scale)'
set ylabel 'imbalance'
set title 'Imbalance/edge cut of various algorithms'

set logscale x 10

plot \
$(
	echo "$ALGORITHMS" | while read -r algorithm
	do
		if [ -z "$algorithm" ]
		then continue
		fi

		file="$tmpdir/$algorithm"
		echo "'$file' title '$algorithm', \\"
	done
)
	 (0.05) with lines title 'Arcswap imbalance threshold'
EOF
