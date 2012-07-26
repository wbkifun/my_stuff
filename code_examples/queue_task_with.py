#!/usr/bin/env python

from threading import Thread, Event
from Queue import Queue
import atexit


class QueueTask(Thread):
    def __init__(s):
        Thread.__init__(s)
        s.daemon = True
        s.queue = Queue()
        s.evt0 = Event()
        s.evt1 = Event()
        atexit.register( s.queue.join )

        s.start()


    def run(s):
        while True:
            func, args, wait_for = s.queue.get()

            if len(wait_for) > 0:
                s.evt0.set()
                for evt in wait_for: evt.wait()
                
            func(*args)

            s.queue.task_done()

    
    def enqueue(s, func, args=[], wait_for=[]):
        s.queue.put((func, args, wait_for))


    def pause(s):
        s.evt0.clear()
        s.evt1.clear()
        s.enqueue(lambda:None, wait_for=[s.evt1])
        s.evt0.wait()


    def resume(s):
        s.evt1.set()



class LockQueueTask:
    def __init__(s, qtask):
        s.qtask = qtask


    def __enter__(s):
        s.qtask.pause()


    def __exit__(s, type, val, traceback):
        s.qtask.resume()




if __name__ == '__main__':
    qtask = QueueTask()
    with LockQueueTask(qtask):
        print('use with')
