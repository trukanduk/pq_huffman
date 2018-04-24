#!/bin/sh
arg=${1:-../out/deep/deep_10K_fast-nn_dist.fvecsl}
gcc -std=c99 read_dists.c -o rd && ./rd $arg
