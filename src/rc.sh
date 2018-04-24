#!/bin/sh

make && ./compute_nn_fast ../../data/deep/deep_10K.fvecs ../out/deep/deep_10K_fast 200 --num-threads 1 --num-dims 2 --num-splits-per-dim 10 --num-blocks-per-dim 11
# make && ./compute_nn_fast ../../data/deep/deep_10K.fvecs ../out/deep/deep_10K_fast 20 --num-threads 1 --num-dims 1 
