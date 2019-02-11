import time

def print_time_info(s, end='\n'):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    t = round(time.time() * 1000)
    # print("%s [%s] %s" % (str(t), times, s))
    print("[%s] %s" % (times, str(s)), end=end)
