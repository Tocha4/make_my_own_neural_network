from multiprocessing import Pool, Process
import time
import os




def f(x):
    if x %2 == 0:
        time.sleep(1)
        print(x)
    else: print(x)
    return x*x

if __name__ == '__main__':
    
    
    start = time.time()
    
    jobs = []
    for i in range(10):
        p = Process(target=f, args=(i,))
        jobs.append(p)
        p.start()

        
    ende = time.time() - start
    print(ende)