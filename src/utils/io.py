# -*- coding: utf8 -*-

import numpy as np
import ynumpy as ynp


def _xvecs_read_header(filename, element_size):
    with open(filename, 'rb') as f:
        num_dimensions_raw = f.read(_FVECS_LINE_HEADER_BYTES)
    num_dimensions = struct.unpack('i', line_size_raw)

    filesize = os.path.getsize()
    line_bytes = _FVECS_LINE_HEADER_BYTES \
                 + num_dimensions * element_size
    num_vectors = int(filesize / line_bytes)
    assert num_vectors * line_bytes == filesize

    return num_vectors, num_dimensions


def _xvecs_read(filename, dtype, sub_dim=None, sub_start=0, limit=None, *,
                batch_size=10000):
    assert not (sub_dim is None and sub_start > 0)

    if sub_dim is None:
        return fvecs_read(filename, limit=limit, dtype=dtype)

    line_bytes = _FVECS_LINE_HEADER_BYTES \
                 + num_dimensions * dtype().itemsize
    num_vectors, num_dimensions = _xvecs_read_header(filename, dtype().itemsize)
    assert sub_start >= 0
    assert sub_start + sub_dim <= num_dimensions

    output = np.zeros(num_vectors, sub_dim, dtype=dtype)

    start_index = 0
    sub_end = sub_start + sub_dim
    with open(filename, 'rb') as f:
        while limit is not None or start_index < limit:
            batch_raw = f.read(line_bytes * batch_size)
            if not len(batch_raw):
                break

            batch = np.frombuffer(batch_raw, dtype=dtype) \
                    .reshape(-1, num_dimensions + 1)
            output[start_index : batch.shape[0]] = \
                    batch[:, sub_start + 1 : sub_end + 1]

            start_index += batch.shape[0]
            if batch.shape[0] < batch_size:
                break

    if limit is None:
        assert start_index == num_vectors
    else:
        assert start_index == limit

    return output


def _limit_to_nmax(limit):
    return limit if limit is not None else -1


def fvecs_read(filename, sub_dim=None, sub_start=0, *,
               dtype=np.float32, limit=None, batch_size=10000):
    if sub_dim is None and sub_start == 0:
        return ynp.fvecs_read(filename, n_max=_limit_to_nmax(limit)) \
                .astype(dtype)

    return _xvecs_read(filename, np.float32, sub_dim, sub_start, limit,
                       batch_size=batch_size).astype(dtype)


def fvecs_write(filename, matrix):
    ynp.fvecs_write(filename, matrix)



def bvecs_read(filename, sub_dim=None, sub_start=0, *,
               light=False, limit=None, batch_size=10000):
    if light:
        assert sub_dim is None
        assert sub_start == 0
        assert limit is None

        return bvecs_read_light(filename)

    if sub_dim is None and sub_start == 0:
        return ynp.bvecs_read(filename, n_max=_limit_to_nmax(limit)) \
                .astype(dtype)

    return _xvecs_read(filename, np.uint8, sub_dim, sub_start, limit,
                       batch_size=batch_size)


def bvecs_write(filename, array, *, light=True):
    if light and array.ndim == 1:
        bvecs_write_light(filename, array)
        return

    assert 1 <= array.ndim <= 2
    ynp.bvecs_write(filename, array.reshape(array.shape[0], -1))


def bvecs_write_light(filename, array):
    array.tofile(filename, array)


def bvecs_read_light(filename):
    return np.fromfile(filename, dtype=np.uint8)
