//
//
// helloflops2
//
// A simple example that gets lots of Flops (Floating Point Operations) on 
// Intel(r) Xeon Phi(tm) co-processors using openmp to scale
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

// dtime 
// 
// returns the current wall clock time
//
double dtime()
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime,(struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return( tseconds );
}

#define FLOPS_ARRAY_SIZE (1024*1024) 
#define MAXFLOPS_ITERS 100000000
#define LOOP_COUNT 128

// number of float pt ops per calculation
#define FLOPSPERCALC 2     
// define some arrays - 
// make sure they are 64 byte aligned
// for best cache access 
float fa[FLOPS_ARRAY_SIZE] __attribute__((align(64)));
float fb[FLOPS_ARRAY_SIZE] __attribute__((align(64)));
//
// Main program - pedal to the metal...calculate using tons o'flops!
// 
int main(int argc, char *argv[] ) 
{
    int i,j,k;
    int numthreads;
    double tstart, tstop, ttime;
    double gflops = 0.0;
    float a=1.1;

    //
    // initialize the compute arrays 

#pragma omp parallel
#pragma omp master
    numthreads = omp_get_num_threads();

    printf("Initializing\r\n");
#pragma omp parallel for
    for(i=0; i<FLOPS_ARRAY_SIZE; i++)
    {
        fa[i] = (float)i + 0.1;
        fb[i] = (float)i + 0.2;
    }	
    printf("Starting Compute on %d threads\r\n",numthreads);

    tstart = dtime();
	
    // scale the calculation across threads requested 
    // need to set environment variables OMP_NUM_THREADS and KMP_AFFINITY

#pragma omp parallel for private(j,k)
    for (i=0; i<numthreads; i++)
    {
        // each thread will work it's own array section
        // calc offset into the right section
        int offset = i*LOOP_COUNT;

        // loop many times to get lots of calculations
        for(j=0; j<MAXFLOPS_ITERS; j++)  
        {
            // scale 1st array and add in the 2nd array 
            for(k=0; k<LOOP_COUNT; k++)  
   	    {
                fa[k+offset] = a * fa[k+offset] + fb[k+offset];
            }
        }
    }
    tstop = dtime();
    // # of gigaflops we just calculated  
    gflops = (double)( 1.0e-9*numthreads*LOOP_COUNT*
                        MAXFLOPS_ITERS*FLOPSPERCALC);    

    //elasped time
    ttime = tstop - tstart;
    //
    // Print the results
    //
    if ((ttime) > 0.0)
    {
        printf("GFlops = %10.3lf, Secs = %10.3lf, GFlops per sec = %10.3lf\r\n",                   gflops, ttime, gflops/ttime);
    }
    return( 0 );
}

