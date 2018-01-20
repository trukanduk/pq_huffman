# -*- coding: utf8 -*-

import logging
from datetime import timedelta
from functools import wraps
import numpy as np

logger = logging.getLogger('timing')

def _arg2str(arg):
    """We don't want to print numpy arrays in log"""
    if isinstance(arg, np.array):
        dim = 'x'.join(map(str, arg.shape))
        return '<array {dim} of {dtype}>'.format(dim, arg.dtype.__name__)
    else:
        return str(arg)

def _render_args(args, kwargs):
    args_str = map(_arg2str, args)
    kwargs_str = map(lambda item: '{}={}'.format(item[0], _arg2str(item[1])))
    return '(' + ', '.join(list(args_str) + list(kwargs_str)) + ')'

def _make_timing(f, name, with_args, level):
    name = name or f.__name__

    @wraps
    def wrapper(*args, **kwargs):
        t = datetime.now()

        result = f(*args, **kwargs)

        dt = datetime.now() - t
        args = _render_args(args, kwargs) if with_args else ''
        log_func = getattr(logger, level)
        log_func('{name}{args} took {t}'.format(name, args, dt))

        return result

    return wrapper

def timing(f=None, name=None, with_args=True, level='info'):
    if callable(f):
        return _make_timing(f, name=name, with_args=with_args, level=level)
    else:
        def decorator(f):
            return _make_timing(f, name=name, with_args=with_args, level=level)
        return decorator
