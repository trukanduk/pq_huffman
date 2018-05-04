#!/usr/bin/env python3

import sys
import utils
import numpy as np
import utils.io as io
import json
import math
from collections import Iterable


def safe_inf(x):
    return x if not math.isinf(x) else 1e100


def calc_main_metrics(dists, prefix):
    n, nn = dists.shape
    result = {
        'num_vectors': n,
        'num_nn': nn,
        'sum_dist': safe_inf(float(dists.sum())),
        'sum_dist_per_vector': safe_inf(float(dists.sum()) / n),
        'sum_dist_at_nn': [safe_inf(float(dists[:,: i + 1].sum())) for i in range(nn)],
        'sum_dist_per_vector_at_nn': [safe_inf(float(dists[:,: i + 1].sum()) / n) for i in range(nn)],
    }
    return {prefix + '_' + k: v for k, v in result.items()}


def safe_div(a, b, zeroval=0.0, eps=1e-5):
    return safe_inf(a / b if abs(b) > eps else zeroval)


def safe_listdiv(a, b, zeroval=0.0, eps=1e-5):
    return [safe_div(ai, bi, zeroval, eps) for ai, bi in zip(a, b)]


def tolist(nparray):
    if not isinstance(nparray, np.ndarray):
        return nparray
    elif nparray.dtype.kind == 'i':
        return list(map(int, list(nparray)))
    elif nparray.dtype.kind == 'f':
        return list(map(float, list(nparray)))
    else:
        return nparray


def calc_diff_main_metrics(result):
    for k in ['sum_dist', 'sum_dist_per_vector']:
        result['diff_' + k] = result['fast_' + k] - result['exact_' + k]
        result['diff_perc_' + k] = safe_div(result['diff_' + k], result['exact_' + k])
        result['fast_perc_' + k] = safe_div(result['fast_' + k], result['exact_' + k])

    for k in ['sum_dist_at_nn', 'sum_dist_per_vector_at_nn']:
        exact = result['exact_' + k]
        fast = result['fast_' + k]
        diff = tolist(np.array(fast) - np.array(exact))
        result['diff_' + k] = tolist(diff)
        result['diff_perc_' + k] = safe_listdiv(tolist(diff), exact)
        result['fast_perc_' + k] = safe_listdiv(fast, exact)


def calc_miss_indices(exact, fast):
    n, nn = exact.shape
    result = np.zeros((nn,), dtype=np.int)
    for i in range(n):
        vec_res = []
        s = set()
        for j in range(nn):
            s.add(exact[i,j])
            s.add(fast[i,j])
            result[j] += len(s) - j - 1

    return {
        'missed_indices_at_nn': tolist(result),
        'missed_indices_per_vec_at_nn': tolist(result / n),
    }


def main():
    exact_template, fast_template = sys.argv[1:]

    indices_exact = io.ivecs_read(exact_template + 'nn_indices.ivecs', light=True)
    dists_exact = io.fvecs_read(exact_template + 'nn_dist.fvecs', light=True)

    indices_fast = io.ivecs_read(fast_template + 'nn_indices.ivecs', light=True)
    dists_fast = io.fvecs_read(fast_template + 'nn_dist.fvecs', light=True)

    result = {}
    result.update(calc_main_metrics(dists_exact, 'exact'))
    result.update(calc_main_metrics(dists_fast, 'fast'))
    calc_diff_main_metrics(result)
    result.update(calc_miss_indices(indices_exact, indices_fast))

    # _dump_types(result)
    # print(result)
    json.dump(result, sys.stdout)
    print()


if __name__ == '__main__':
    main()
