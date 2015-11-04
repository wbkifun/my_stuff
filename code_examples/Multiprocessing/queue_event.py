import numpy as np
import multiprocessing as mp
import time



class Device(mp.Process):
    def __init__(self):
        Process.__init__(self)
        self.task_queue = mp.JoinableQueue()
        self.event = mp.Event()


    def run(self):
        while True:
            func, waits, args = self.task_queue.get()
            if func == 'exit':
                self.task_queue.task_done()
                break
            else:
                self.event.clear()              # set False
                for evt in waits: evt.wait()    # wait for prerequisite conditions
                func(*args)
                self.event.set()                # set True



class Code
class SomeTask(object):
    def __init__(self, n, a):
        self.arr = np.arange(n)



class Platform(object):
    def __init__(self, numdev):
        self.devs = [Device() for i in xrange(numdev)]
        self.tasks = list()


    def set(self):
        for task in self.tasks:
            task.set()


    def run(self):
        for task in self.tasks:
            task.run()


    def finalize(self):
        for task in self.tasks:
            task.finalize()



class Task(object):
    def __init__(self, platform):
        pass



if __name__ == '__main__':
    platform = Platform(5)
    Task('task1', platform, variables, )
    Task(platform, )
    Task(platform, )

    platform.set()
    platform.run()
    platform.finalize()
