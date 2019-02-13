import time


def print_time_info(string, end='\n', print_error=False):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    # t = round(time.time() * 1000)
    # print("%s [%s] %s" % (str(t), times, s))
    string = "[%s] %s" % (times, str(string))
    if print_error:
        print(len(string) * '-')
    print(string, end=end)
