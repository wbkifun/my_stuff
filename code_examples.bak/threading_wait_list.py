#!/usr/bin/env python

from threading import Thread
from time import sleep, time
import numpy as np

def delay(t, name, t0):
	t1 = time() - t0
	sleep(t)
	t2 = time() - t0
	print('%s %d\t%1.2f\t%1.2f' % (name, t, t1, t2))


nx = 10
ts = np.random.randint(0, 10, nx)
names = [chr(num) for num in (np.arange(nx) + ord('a'))]
print(names)
print(ts)
wait_lists = [[], [], [0], [], [1], [], [], [2, 5], [], []]

'''
print('sequential : predict %d s' % ts.sum())
t0 = time()
for t, name in zip(ts, names, t0):
	delay(t, name)
print('sequential : result %f s' % (time() - t0))
'''

'''
print('threading : predict %d s' % ts.max())
t0 = time()
threads = []
for t, name in zip(ts, names):
	th = Thread(target=delay, args=(t, name, t0))
	th.start()
	threads.append(th)

for thread in threads:
	thread.join()
print('threading : result %f s' % (time() - t0))
'''

'''
# Try 1 (fail)
def run_as_thread(func, args, wait_thread):
	for th in wait_thread: th.join()
	th = Thread(target=func, args=args)
	th.start()
	return th

t_max = max(ts.max(), ts[0]+ts[2], ts[1]+ts[4], max(ts[0]+ts[2],ts[5]) + ts[7])
print('wait_list : predict %d s' % t_max)
t0 = time()
threads = []
for t, name, wait_list in zip(ts, names, wait_lists):
	wait_thread = [threads[i] for i in wait_list]
	th = run_as_thread(delay, (t, name, t0), wait_thread)
	threads.append(th)

for thread in threads:
	thread.join()
print('threading : result %f s' % (time() - t0))
'''



# Try 2 ()
class ThreadWait(Thread):
	def __init__(s, func, args, wait_thread):
		s.wait_thread = wait_thread
		Thread.__init__(s, target=func, args=args)

	def start(s):
		for th in s.wait_thread: th.join()
		Thread.start(s)

t_max = max(ts.max(), ts[0]+ts[2], ts[1]+ts[4], max(ts[0]+ts[2],ts[5]) + ts[7])
print('wait_list : predict %d s' % t_max)
t0 = time()
threads = []
for t, name, wait_list in zip(ts, names, wait_lists):
	wait_thread = [threads[i] for i in wait_list]
	th = ThreadWait(delay, (t, name, t0), wait_thread)
	th.start()
	threads.append(th)

for thread in threads:
	thread.join()
print('threading : result %f s' % (time() - t0))
