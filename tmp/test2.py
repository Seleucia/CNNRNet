import multiprocessing
from multiprocessing.pool import ThreadPool
import time

data = (
    ['a', '2'], ['b', '4'], ['c', '6'], ['d', '8'],
    ['e', '1'], ['f', '3'], ['g', '5'], ['h', '7']
)

def mp_worker((inputs, the_time)):
    print " Processs %s\tWaiting %s seconds" % (inputs, the_time)
    time.sleep(int(the_time))
    print " Process %s\tDONE" % inputs
    return the_time

def mp_handler():
    p = ThreadPool(100)
    return p.map(mp_worker, data)

if __name__ == '__main__':
   res= mp_handler()
   print "completed"