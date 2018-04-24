#!/usr/bin/env python
# -*- coding: utf8 -*-

from datetime import datetime
from multiprocessing import cpu_count
import argparse
import logging
import numpy as np
import sys
import ynumpy as ynp

import utils
from utils.timing import timing
import utils.io as io


def make_exact_nn(input_filename, output_template, num_nn, num_threads=None,
                  light_indices=False, light_dist=False):
    num_threads = utils.make_num_threads(num_threads)

    data = io.fvecs_read(input_filename)
    nn_indices, nn_dist = ynp.knn(data, data, nnn=num_nn + 1, nt=num_threads)
    nn_indices = nn_indices[:, 1:].copy() # TODO: make bunchs fvecs_write
    nn_dist = nn_dist[:, 1:].copy() # TODO:

    io.mkdirs(output_template)
    io.ivecs_write(output_template + 'nn_indices.ivecs', nn_indices,
                   light=light_indices)
    io.fvecs_write(output_template + 'nn_dist.fvecs', nn_dist,
                   light=light_dist)


@timing
def make_nn(input_filename, output_template, num_nn, method='exact',
            num_threads=None, light_indices=False, light_dist=False):
    if method == 'exact':
        make_exact_nn(input_filename, output_template, num_nn,
                      num_threads=num_threads, light_indices=light_indices,
                      light_dist=light_dist)
    else:
        raise NotImplementedError('Unsupported method "{}"'.format(method))

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='run product quantization')
    parser.add_argument('input', help='Input data file')
    parser.add_argument('output_template', help='''
        Output files template. Filenames will be
        <template>-(nn_indices|nn_dist).(i|f)vecs(l)?
    ''')
    parser.add_argument('num_nn', type=int, help='Num nearest_neighbors')
    parser.add_argument('--method', default='exact',
                        help='method: exact (default)')
    parser.add_argument('--exact', action='store_const', dest='method',
                        const='exact', help='Use exact method')
    parser.add_argument('--num-threads', type=int, default=None,
                        help='Num threads to run (default is num_cpu + 1)')
    parser.add_argument('--light-indices', action='store_true', default=False,
                        help='save indices in light format')
    parser.add_argument('--no-light-indices', action='store_false',
                        dest='light_indices',
                        help='save indices in non-light format')
    parser.add_argument('--light-dist', action='store_true', default=False,
                        help='save distances in light format')
    parser.add_argument('--no-light-dist', action='store_false',
                        dest='light_dist',
                        help='save distances in non-light format')

    args = parser.parse_args()
    make_nn(args.input, args.output_template, args.num_nn, method=args.method,
            num_threads=args.num_threads, light_indices=args.light_indices,
            light_dist=args.light_dist)

if __name__ == '__main__':
    main()
