#!/bin/bash

PQ_HOME=${PQ_HOME:-$HOME/pq}

nn_dir=$1

if [ "$nn_dir" = 'all' ]
then
    for nn_dir_ in $(ls "$PQ_HOME/out/nn")
    do
        echo $nn_dir_
        bash $0 $nn_dir_
    done
    exit 0
elif [ ! "$nn_dir" ]
then
    echo "Usage: $0 <nn output dir name>" >&2
    exit 1
fi

DIR="$PQ_HOME/out/nn/$nn_dir"
if [ ! -d "$DIR" ]
then
    echo "No $DIR dir" >&2
    exit 1
fi

cd "$DIR"

fix_nn() {
    if [ ! -f "$1" ]
    then
        echo "No file $DIR/$1" >&2
        return
    fi

    fsize=$(ls -l "$1" | awk '{print $5}')
    fsize2=$(expr $fsize % 10)

    if [ "$fsize2" != '8' ]
    then
        echo "Already fixed $1" >&2
        return
    fi

    mv "$1" "${1}_"
    tail -c +9 "${1}_" > "$1"
}

fix_nn "nn_indices.ivecsl"
fix_nn "nn_dist.fvecsl"
