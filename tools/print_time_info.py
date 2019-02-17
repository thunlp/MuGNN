import time


def print_time_info(string, end='\n', dash_top=False, dash_bot=False):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    string = "[%s] %s" % (times, str(string))
    if dash_top:
        print(len(string) * '-')
    print(string, end=end)
    if dash_bot:
        print(len(string) * '-')
