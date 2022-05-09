# num-part

A quality evaluation framework for number-partitioning algorithms.

This program generates sets of random numbers that follow a given distribution,
runs algorithms on these sets, then saves the results in a SQLite database.

## Usage

See `num-part --help` for usage instructions.

The syntax for algorithms is the same as `mesh-part(1)`. However, only number
partitioning algorithms will work.

After some runs, you can query the SQLite database for results. For example,

```
sqlite3 num_part.db <<EOF
SELECT imbalance FROM experiment
WHERE algorithm = 'greedy,2'
EOF
```

will return the imbalance of all runs of *greedy* (2 parts).
