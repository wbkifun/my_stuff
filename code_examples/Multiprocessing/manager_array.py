import numpy as np
import multiprocessing




def worker(arr, idx, value):
        arr[idx] = value




if __name__ == '__main__':
    nproc = 10
    mgr = multiprocessing.Manager()
    arr = mgr.Array('f', np.zeros(nproc))
    jobs = [ multiprocessing.Process(target=worker, args=(arr, i, i*2)) for i in range(nproc) ] 
    
    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    print 'Results:', arr
