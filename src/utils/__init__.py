from .timing import *

from multiprocessing import cpu_count

def make_num_threads(num_threads=None, add_if_max=1):
    if num_threads is None or num_threads < 0:
        return cpu_count() + add_if_max
    return num_threads
