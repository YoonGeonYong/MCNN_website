import threading
from datetime import datetime


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t


def i_wanna_interval():
    print(datetime.now(), "Hi")


set_interval(i_wanna_interval, 10)