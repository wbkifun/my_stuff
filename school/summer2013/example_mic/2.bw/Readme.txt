$ icc -mmic triad_omp_bw.c -DBW -openmp
$ micnativeloadex ./a.out

ITER=1000 SIZE=1000 MaxThreads=224 

-------> BandWidht <-----------------
 Time      :     0.006595 (sec)
 MemSize   :     0.000016 (GB) 
 GBYTE     :     0.016000 (GB) 
 GBYTE/s   :     2.426031 (GB/s) 

