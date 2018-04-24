#!/usr/bin/env python3

import sys
import utils
import numpy as np
import utils.io as io
import json
import math


def safe_inf(x):
    return x if not math.isinf(x) else 1e100


def calc_main_metrics(dists, prefix):
    n, nn = dists.shape
    result = {
        'num_vectors': n,
        'num_nn': nn,
        'sum_dist': safe_inf(dists.sum()),
        'sum_dist_per_vector': safe_inf(dists.sum() / n),
        'sum_dist_at_nn': [safe_inf(dists[:,: i + 1].sum()) for i in range(nn)],
        'sum_dist_per_vector_at_nn': [safe_inf(dists[:,: i + 1].sum() / n) for i in range(nn)],
    }
    return {prefix + '_' + k: v for k, v in result.items()}


def safe_div(a, b, zeroval=0.0, eps=1e-5):
    return safe_inf(a / b if abs(b) > eps else zeroval)


def safe_listdiv(a, b, zeroval=0.0, eps=1e-5):
    return [safe_div(ai, bi, zeroval, eps) for ai, bi in zip(a, b)]


def calc_diff_main_metrics(result):
    for k in ['sum_dist', 'sum_dist_per_vector']:
        result['diff_' + k] = result['fast_' + k] - result['exact_' + k]
        result['diff_perc_' + k] = safe_div(result['diff_' + k], result['exact_' + k])
        result['fast_perc_' + k] = safe_div(result['fast_' + k], result['exact_' + k])

    for k in ['sum_dist_at_nn', 'sum_dist_per_vector_at_nn']:
        exact = result['exact_' + k]
        fast = result['fast_' + k]
        diff = np.array(fast) - np.array(exact)
        result['diff_' + k] = list(diff)
        result['diff_perc_' + k] = safe_listdiv(list(diff), exact)
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
        'missed_indices_at_nn': list(result),
        'missed_indices_per_vec_at_nn': list(result / n),
    }


def main():
    exact_template, fast_template = sys.argv[1:]

    indices_exact = io.ivecs_read(exact_template + 'nn_indices.ivecs', light=True)
    dists_exact = io.fvecs_read(exact_template + 'nn_dist.fvecs', light=True)

    indices_fast = io.lvecs_read(fast_template + 'nn_indices.lvecs', light=True)
    dists_fast = io.fvecs_read(fast_template + 'nn_dist.fvecs', light=True)

    result = {}
    result.update(calc_main_metrics(dists_exact, 'exact'))
    result.update(calc_main_metrics(dists_fast, 'fast'))
    calc_diff_main_metrics(result)
    result.update(calc_miss_indices(indices_exact, indices_fast))

    print(result)
    # json.dump(result, sys.stdout)


if __name__ == '__main__':
    main()
