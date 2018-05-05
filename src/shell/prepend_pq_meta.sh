#!/bin/bash

N_sift100K=100000
N_sift1M=1000000
N_deep100K=100000
N_deep1M=1000000
N_deep10M=10000000
N_deep350M=358480000

D_sift100K=128
D_sift1M=128
D_deep100K=96
D_deep1M=96
D_deep10M=96
D_deep350M=96

make prepend_vecsl_meta || exit 1
for dataset in sift100K sift1M deep100K deep1M deep10M deep350M
do
    for m in 4 8 16 32
    do
        n_name="N_$dataset"
        num_vectors=${!n_name}
        d_name="D_$dataset"
        dim=${!d_name}
        dim_star=$(expr $dim / $m)

        DIR="$PQ_HOME/out/pq/${dataset}_$m"
        if [ ! -f "$DIR/pq_indices.bvecsl" ]
        then
            echo "Skip $dataset $m ($DIR)"
            continue
        fi
        ./prepend_vecsl_meta "$DIR/pq_indices.bvecsl" $num_vectors $m || exit 1
        ./prepend_vecsl_meta "$DIR/pq_centroids.fvecsl" $(expr 256 \* $m) $dim_star || exit 1
        echo "$dataset $m OK"
    done
done
