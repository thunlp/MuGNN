import time
import functools


def print_time_info(string, end='\n', dash_top=False, dash_bot=False, file=None):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    string = "[%s] %s" % (times, str(string))
    if dash_top:
        print(len(string) * '-', file=file)
    print(string, end=end, file=file)
    if dash_bot:
        print(len(string) * '-', file=file)


def timeit(func):
    @functools.wraps(func)
    def timed(*args, **kw):
        ts = time.time()
        print_time_info('Method: %s started!' %(func.__name__), dash_top=True)
        result = func(*args, **kw)
        te = time.time()
        print_time_info('Method: %s cost %.2f sec!' %(func.__name__, te-ts), dash_bot=True)
        return result
    return timed