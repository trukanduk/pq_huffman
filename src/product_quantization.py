#!/usr/bin/env python
# -*- coding: utf8 -*-

from datetime import datetime
from multiprocessing import cpu_count
import argparse
import logging
import numpy as np
import sys
# FIXME:
if sys.version[0] == '2':
    sys.path = ['/home/ilya/yael/yael_v438_2/yael'] + sys.path
else:
    sys.path = ['/home/ilya/yael/yael_v438/yael'] + sys.path

import struct
import ynumpy as ynp

from utils.timing import timing
import utils.io as io


def _determine_index_dtype(num_centroids):
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if 2**(8 * dtype().itemsize) >= num_centroids:
            return dtype
    else:
        raise ValueError("Too much num_centroids")


NUM_CENTROIDS = 2**8
KMEANS_INIT = 'kmeans++'


# @timing
def product_quantization_part(subvectors, num_centroids=NUM_CENTROIDS,
                              num_threads=None, kmeans_init=KMEANS_INIT):
    num_vectors, num_dimensions_part = subvectors.shape
    if num_threads is None:
        num_threads = cpu_count() + 1

    centroids, _, _, indices, _ = ynp.kmeans(subvectors, num_centroids,
                                              nt=num_threads, output=None,
                                              init=kmeans_init)
    return centroids, indices


# @timing
def product_quantization(input_filename, output_filename_template, num_parts,
                         num_centroids=NUM_CENTROIDS, num_threads=None,
                         kmeans_init=KMEANS_INIT, light_indices=True,
                         return_centroids=True, return_indices=True):
    num_vectors, num_dimensions = io.fvecs_read_header(input_filename)
    num_dimensions_part = int(num_dimensions / num_parts)
    assert num_dimensions_part * num_parts == num_dimensions

    io.mkdirs(output_filename_template)
    indices_template = '{}-indices-{:02}-{{:02}}.ivecs' \
            .format(output_filename_template, num_parts)
    centroids_template = '{}-centroids-{:02}-{{:02}}.fvecs' \
            .format(output_filename_template, num_parts)

    centroids_result, indicecs_result = [], []
    for part_index in range(num_parts):
        subvectors = io.fvecs_read(input_filename, num_dimensions_part,
                                   part_index * num_dimensions_part)
        centroids, indices = \
                product_quantization_part(subvectors, num_centroids,
                                          num_threads=num_threads,
                                          kmeans_init=kmeans_init)
        io.fvecs_write(centroids_template.format(part_index), centroids)
        io.bvecs_write(indices_template.format(part_index), indices,
                       light=light_indices)

        if return_centroids:
            centroids_result.append(centroids)
        if return_indices:
            indicecs_result.append(indices)

        del centroids
        del indices

    result = ([np.array(centroids_result)] if return_centroids else []) \
             + ([np.array(indices_result)] if return_indices else [])
    if result:
        return tuple(result)


def main():
    parser = argparse.ArgumentParser(description='run product quantization')
    parser.add_argument('input', help='Input data file')
    parser.add_argument('output_template', help='''
        Output files template. Filenames will be
        <template>-(indices|centroids)-<num parts>-<part num>.(f|i)vecs
    ''')
    parser.add_argument('num_parts', type=int, help='Num parts to split')
    parser.add_argument('--num-centroids', type=int, default=NUM_CENTROIDS,
                        help='Num centroids to create (default is 256)')
    parser.add_argument('--num-threads', type=int, default=None,
                        help='Num threads to run (default is num_cpu + 1)')
    parser.add_argument('--kmeans-init', type=str, default=KMEANS_INIT,
                        help='Num threads to run: "random" or "kmeans++" '
                                '(default is {})'.format(KMEANS_INIT))
    parser.add_argument('--light-indices', action='store_true',
                        default=True,
                        help='Store indices in light format (default)')
    parser.add_argument('--no-light-indices', action='store_false',
                        help='Store indices in non-light format')

    args = parser.parse_args()
    print(args)
    product_quantization(args.input, args.output_template, args.num_parts,
                         num_centroids=args.num_centroids,
                         num_threads=args.num_threads,
                         light_indices=args.light_indices,
                         return_centroids=False,
                         return_indices=False)

if __name__ == '__main__':
    main()
