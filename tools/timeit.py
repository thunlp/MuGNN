import time
import functools
from .print_time_info import print_time_info

def timeit(func):
    @functools.wraps(func)
    def timed(*args, **kw):
        ts = time.time()
        print_time_info('Method: %s started!' %(func.__name__))
        result = func(*args, **kw)
        te = time.time()
        print_time_info('Method: %s cost %.2f sec!' %(func.__name__, te-ts))
        return result
    return timed