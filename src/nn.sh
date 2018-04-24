#!/bin/bash

PQ_HOME=${PQ_HOME:-$HOME/pq}

print_help() {
    echo "Usage NUM_NN=100 NUM_THREADS=4 ./$0 exact <dataset>" >&2
    echo "      $0 fast <dataset> <num-dimensions> <num-blocks> <num-splits> <num-threads>" >&2
    exit 1
}

now_iso() {
    python3 -c 'import datetime; print(datetime.datetime.now().isoformat())'
}

diff_iso() {
    echo $1 | python3 -c 'import datetime; print(datetime.datetime.now() - datetime.datetime.fromtimestamp(float(input())))'
}

now_ts() {
    python3 -c 'import time; print(time.time())'
}

diff_ts() {
    echo $1 | python3 -c 'import time; print(time.time() - float(input()))'
}

nn_type=$1
dataset=$2

if [ "$nn_type" = 'exact' ]
then
    NUM_NN=${NUM_NN:-50}
    NUM_THREADS=${NUM_THREADS:-1}

    OUT_DIR="$PQ_HOME/out/nn/${dataset}_${NUM_NN}"
    mkdir -p $OUT_DIR

    echo "Starting exact $dataset with NUM_NN=$NUM_NN, NUM_THREADS=$NUM_THREADS at $(now_iso)"

    start=$(now_ts)
    python3 compute_nn.py "$PQ_HOME/data/${dataset}.fvecs" "$OUT_DIR/" ${NUM_NN:-50} --num-threads ${NUM_THREADS:-1} --light-indices --light-dist || exit 1
    t=$(diff_ts $start)
    echo $t > "$OUT_DIR/time_${NUM_THREADS}"

    echo "    Done in $(diff_iso $start)"

elif [ "$nn_type" = 'fast' ]
then
    echo "TODO" >&2
    print_help

else
    echo "Unknown NN type: '$nn_type'" >&2
    print_help
fi
