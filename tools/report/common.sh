ALGORITHMS="
-a kk,2
-a kk,2 -a fm,0.05
-a kk,2 -a fm,0.05,16
-a rcb,1
-a rcb,1 -a fm,0.05
-a hilbert,2
-a hilbert,2 -a fm,0.05

-a kk,3
-a kk,3 -a fm,0.05
-a hilbert,3

-a kk,4
-a rcb,2
-a hilbert,4

-a kk,128
-a rcb,7
-a hilbert,128
"

WEIGHT_DISTRIBUTIONS="
constant,1
linear,x,0,100
spike,1000
"

say() {
	verb=$1
	shift

	YELLOW="\e[33;1m"
	RESET="\e[0m"
	if ! [ -t 2 ]
	then
		# Not a tty, dont use escape codes
		YELLOW=""
		RESET=""
	fi

	MAX_VERB_LEN=12
	verblen=${#verb}
	indent=$((MAX_VERB_LEN - verblen))

	printf "%${indent}s$YELLOW%s$RESET %s\n" "" "$verb" "$@" >&2
}

sayfile() {
	verb=$1
	file=$2

	say "$verb" "$(basename "$file")"
}

command -v jq >/dev/null || {
	echo "command 'jq' not found" >&2
	exit 1
}

TARGET_DIR=$(cargo metadata --format-version=1 | jq -r '.target_directory')
