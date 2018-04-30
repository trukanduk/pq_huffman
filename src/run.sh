#!/bin/bash

PQ_HOME=${PQ_HOME:-$HOME/pq}

print_help() {
    echo "Usage NUM_NN=100 NUM_THREADS=4 ./$0 nn-exact <dataset>" >&2
    echo "      NUM_NN=100 NUM_THREADS=4 NUM_DIM=3 NUM_SPLITS=10 NUM_BLOSKS=11 ./$0 nn-fast <dataset>" >&2
    echo "      M=8 NUM_THREADS=4 ./$0 pq <dataset>" >&2
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

action=$1
dataset=$2

if [ "$action" = 'nn-exact' ]
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

elif [ "$action" = 'nn-fast' ]
then
    make compute_nn_fast || exit 1

    NUM_NN=${NUM_NN:-50}
    NUM_THREADS=${NUM_THREADS:-1}

    NUM_DIM=${NUM_DIM:-3}
    NUM_SPLITS=${NUM_SPLITS:-10}
    NUM_BLOCKS=${NUM_BLOCKS:-11}

    OUT_DIR="$PQ_HOME/out/nn/${dataset}_${NUM_NN}_d${NUM_DIM}_s${NUM_SPLITS}_b${NUM_BLOCKS}"
    mkdir -p $OUT_DIR

    echo "Starting fast $dataset with NUM_NN=$NUM_NN, NUM_THREADS=$NUM_THREADS, NUM_DIM=${NUM_DIM}x${NUM_BLOCKS}/${NUM_SPLITS} at $(now_iso)..."

    start=$(now_ts)
    ./compute_nn_fast "$PQ_HOME/data/${dataset}.fvecs" "$OUT_DIR/" $NUM_NN \
            --with-blocks-stat --num-threads $NUM_THREADS --num-dims $NUM_DIM \
            --num-splits-per-dim $NUM_SPLITS --num-blocks-per-dim $NUM_BLOCKS > "$OUT_DIR/stdout.log" 2> "$OUT_DIR/stderr.log" || exit 1
    t=$(diff_ts $start)
    echo $t >> "$OUT_DIR/time_${NUM_THREADS}"

    echo "    Done nn in $(diff_iso $start). Starting compare..."
    EXACT_OUT="$PQ_HOME/out/nn/${dataset}_${NUM_NN}"
    python3 compare_nn.py "$EXACT_OUT/" "$OUT_DIR/" >> $OUT_DIR/diff.json
    echo "    Done in $(diff_iso $start)"

elif [ "$action" = 'pq' ]
then
    make pq || exit 1

    M=${M:-8}
    NUM_THREADS=${NUM_THREADS:-1}

    OUT_DIR="$PQ_HOME/out/pq/${dataset}_${M}"
    mkdir -p $OUT_DIR

    echo "Starting fast $dataset with M=$M, NUM_THREADS=$NUM_THREADS at $(now_iso)..."

    start=$(now_ts)
    ./pq "$PQ_HOME/data/${dataset}.fvecs" "$OUT_DIR/" $M \
            --num-threads $NUM_THREADS --compute-error # > "$OUT_DIR/stdout.log" 2> "$OUT_DIR/stderr.log" || exit 1
    t=$(diff_ts $start)
    echo $t >> "$OUT_DIR/time_${NUM_THREADS}"
    echo "    Done in $(diff_iso $start)"

else
    echo "Unknown action: '$action'" >&2
    print_help
fi
