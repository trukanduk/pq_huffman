#!/bin/sh

export NUM_THREADS=16
for m in 4 8 16 32
do
    M=$m ./run.sh pq deep10M
done

