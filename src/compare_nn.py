#!/usr/bin/env python3

import sys
import utils
import numpy as np

if sys.version[0] == '2':
    sys.path = ['/home/ilya/yael/yael_v438_2/yael'] + sys.path
else:
    sys.path = ['/home/ilya/yael/yael_v438/yael'] + sys.path

import utils.io as io

def main():
    exact_template, fast_template = sys.argv[1:]

    indices_exact = io.ivecs_read(exact_template + '-nn_indices.ivecs', light=True)
    dists_exact = io.fvecs_read(exact_template + '-nn_dist.fvecs', light=True)

    indices_fast = io.lvecs_read(fast_template + '-nn_indices.lvecs', light=True)
    dists_fast = io.fvecs_read(fast_template + '-nn_dist.fvecs', light=True)

    print(dists_fast[0])
    print(((dists_exact - dists_fast)**2).sum(), np.abs(dists_exact).sum())


if __name__ == '__main__':
    main()
