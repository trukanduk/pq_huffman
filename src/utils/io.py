# -*- coding: utf8 -*-

import numpy as np
import ynumpy as ynp

import struct
import os
import os.path

_FVECS_LINE_HEADER_BYTES = 4


def _xvecs_read_header(filename, dtype, light=False):
    if light:
        return _xvecs_light_read_header(filename)

    with open(filename, 'rb') as f:
        num_dimensions_raw = f.read(_FVECS_LINE_HEADER_BYTES)
    num_dimensions = struct.unpack('i', num_dimensions_raw)[0]

    filesize = os.path.getsize(filename)
    line_bytes = _FVECS_LINE_HEADER_BYTES \
                 + num_dimensions * dtype().itemsize
    num_vectors = int(filesize / line_bytes)
    assert num_vectors * line_bytes == filesize

    return num_vectors, num_dimensions


def _xvecs_read(filename, dtype, sub_dim=None, sub_start=0, limit=None,
                batch_size=10000):
    assert not (sub_dim is None and sub_start > 0)

    if sub_dim is None:
        return fvecs_read(filename, limit=limit, dtype=dtype)

    num_vectors, num_dimensions = _xvecs_read_header(
            filename, dtype, light=False)
    line_bytes = _FVECS_LINE_HEADER_BYTES \
                 + num_dimensions * dtype().itemsize
    assert sub_start >= 0
    assert sub_start + sub_dim <= num_dimensions

    output = np.zeros((num_vectors, sub_dim), dtype=dtype)

    start_index = 0
    num_elements_per_header = int(4 / dtype().itemsize)
    assert num_elements_per_header > 0
    sub_start += num_elements_per_header
    sub_end = sub_start + sub_dim
    with open(filename, 'rb') as f:
        while limit is None or start_index < limit:
            batch_raw = f.read(line_bytes * batch_size)
            if not len(batch_raw):
                break

            batch = np.frombuffer(batch_raw, dtype=dtype) \
                    .reshape(-1, num_dimensions + num_elements_per_header)
            output[start_index : start_index + batch.shape[0]] = \
                    batch[:, sub_start : sub_end]

            start_index += batch.shape[0]
            if batch.shape[0] < batch_size:
                break

    if limit is None:
        assert start_index == num_vectors
    else:
        assert start_index == limit

    return output


XVECS_LIGHT_HEADER_SIZE = 4 * 2


def _xvecs_light_read_header(filename):
    with open(filename + 'l', 'rb') as f:
        header_raw = f.read(XVECS_LIGHT_HEADER_SIZE)
    return struct.unpack('ii', header_raw)


def _xvecs_light_read(filename, dtype, limit=None):
    num_vectors, num_dimensions = _xvecs_light_read_header(filename)
    with open(filename + 'l', 'rb') as f:
        f.read(XVECS_LIGHT_HEADER_SIZE)
        result = np.fromfile(f, dtype=dtype, count=_limit_to_nmax(limit))
        return result.reshape(num_vectors, num_dimensions)


def _xvecs_light_write(filename, matrix):
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)


    with open(filename + 'l', 'wb') as f:
        f.write(struct.pack('ii', *matrix.shape))
        matrix.tofile(f)


def _limit_to_nmax(limit):
    return limit if limit is not None else -1


def fvecs_read(filename, sub_dim=None, sub_start=0,
               dtype=np.float32, limit=None, batch_size=10000, light=False):
    if light:
        assert sub_dim is None
        assert sub_start == 0

        return _xvecs_light_read(filename, dtype=np.float32, limit=limit) \
                .astype(dtype)

    if sub_dim is None and sub_start == 0:
        return ynp.fvecs_read(filename, nmax=_limit_to_nmax(limit)) \
                .astype(dtype)

    return _xvecs_read(filename, np.float32, sub_dim, sub_start, limit,
                       batch_size=batch_size).astype(dtype)


def fvecs_read_header(filename, light=False):
    return _xvecs_read_header(filename, np.float32, light=light)


def fvecs_write(filename, matrix, light=False):
    if light:
        _xvecs_light_write(filename, matrix.astype(np.float32))
    else:
        ynp.fvecs_write(filename, matrix)


def ivecs_read_header(filename):
    return _xvecs_read_header(filename, np.int32)


def ivecs_read(filename, sub_dim=None, sub_start=0, dtype=np.int32,
               limit=None, batch_size=10000, light=False):
    if light:
        assert sub_dim is None
        assert sub_start == 0

        return _xvecs_light_read(filename, dtype=np.int32, limit=limit) \
                .astype(dtype)

    if sub_dim is None and sub_start == 0:
        return ynp.ivecs_read(filename, nmax=_limit_to_nmax(limit)) \
                .astype(dtype)

    return _xvecs_read(filename, np.int32, sub_dim, sub_start, limit,
                       batch_size=batch_size).astype(dtype)


def lvecs_read_header(filename):
    return _xvecs_read_header(filename, np.int64)


def lvecs_read(filename, sub_dim=None, sub_start=0, dtype=np.int64,
               limit=None, batch_size=10000, light=False):
    if light:
        assert sub_dim is None
        assert sub_start == 0

        return _xvecs_light_read(filename, dtype=np.int64, limit=limit) \
                .astype(dtype)

    if sub_dim is None and sub_start == 0:
        return ynp.ivecs_read(filename, nmax=_limit_to_nmax(limit)) \
                .astype(dtype)

    return _xvecs_read(filename, np.int64, sub_dim, sub_start, limit,
                       batch_size=batch_size).astype(dtype)


def ivecs_write(filename, matrix, light=False):
    if light:
        _xvecs_light_write(filename, matrix.astype(np.int32))
        return

    # TODO: bunchs write
    matrix_ = np.zeros((matrix.shape[0], matrix.shape[1] + 1), dtype=np.int32)
    matrix_[:0] = matrix.shape[1]
    matrix_[:, 1:] = matrix
    matrix_.tofile(filename)


def bvecs_read(filename, sub_dim=None, sub_start=0, dtype=np.uint8,
               light=False, limit=None, batch_size=10000):
    if light:
        assert sub_dim is None
        assert sub_start == 0

        return _xvecs_light_read(filename, dtype=np.uint8, limit=limit) \
                .astype(dtype)

    if sub_dim is None and sub_start == 0:
        return ynp.bvecs_read(filename, nmax=_limit_to_nmax(limit)) \
                .astype(dtype)

    return _xvecs_read(filename, np.uint8, sub_dim, sub_start, limit,
                       batch_size=batch_size)


def bvecs_write(filename, array, light=True):
    if light or array.ndim == 1:
        _xvecs_light_write(filename, array.astype(np.uint8))
        return

    assert 1 <= array.ndim <= 2
    ynp.bvecs_write(filename, array.reshape(array.shape[0], -1))


def mkdirs(filepath):
    dirpath, _ = os.path.split(filepath)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
