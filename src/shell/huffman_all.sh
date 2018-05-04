#!/bin/bash

for dataset in sift100K deep100K sift1M deep1M deep350M
do
    for m in 4 8 16 32
    do
        export M=$m
	SORT=sort CONTEXT=1 ./run.sh huffman $dataset
	INPUT=sort CONTEXT=1 ./run.sh huffman-decode $dataset
	SORT=nosort CONTEXT=0 ./run.sh huffman $dataset
    done 
done

