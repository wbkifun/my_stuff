#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h> 
#define real double

#define MaxIter  1000

#define MaxArray (1000)

double dtime()
{
	
	double t = 0.0;
	
	struct timeval timev;
	gettimeofday(&timev, NULL);
	t = (double)(timev.tv_sec + timev.tv_usec*1e-6);
	return t;
}

#ifdef FLOP
void triad_flop(real* A, real* B, real* C,  int iter, int length );
#endif

#ifdef BW
void triad_byte(real* A, real* B, int iter, int length );
#endif

int main (int argc, char** argv)
{ 
    real time;
	
	int i,j;

	double sTime, eTime, tTime;
	
	int iter, size, nmax;

	size = MaxArray;
		
	iter = MaxIter;

	nmax= omp_get_max_threads();

	printf("\nITER=%d SIZE=%d MaxThreads=%d \n",iter, size, nmax);
	
    real *a, *b, *c;

	a = (real*) _mm_malloc(sizeof(real)*size,64);
	b = (real*) _mm_malloc(sizeof(real)*size,64);
	c = (real*) _mm_malloc(sizeof(real)*size,64);

#pragma omp parallel for
	for ( i=0;i<size; i++)
	{
		a[i] = (real)i + 1.0;
		b[i] = (real)i + 1.1;
		c[i] = (real)i + 1.2;
	}

#ifdef FLOP
	sTime = dtime();
	triad_flop(a,b,c,iter,size);
	eTime = dtime();
	tTime_flop = eTime - sTime;
	printf("\n-------> Performance on Phi <-----------------\n");
    printf(" Time      : %12.6lf (sec)\n", tTime_flop);
	double GFLOPS;
	GFLOPS = (1.0e-09 * iter * size * 2.0)/tTime_flop;
	printf(" Gflops    : %12.6lf (GF/s) \n",GFLOPS);
#endif
#ifdef BW
	double tTime_bw;	
	sTime = dtime();
	triad_byte(a,b,iter,size);
	eTime = dtime();
	tTime_bw = eTime - sTime;

	printf("\n-------> BandWidht <-----------------\n");
    printf(" Time      : %12.6lf (sec)\n", tTime_bw);
	printf(" MemSize   : %12.6lf (GB) \n", 2.0 * size * sizeof(real) *1e-9);
	double GBYTES;
	GBYTES = (1.0e-09 * iter * size * 2 * sizeof(real));

	printf(" GBYTE     : %12.6lf (GB) \n",GBYTES);
	printf(" GBYTE/s   : %12.6lf (GB/s) \n",GBYTES/tTime_bw);
#endif
	

	_mm_free(a);
	_mm_free(b);
	_mm_free(c);

    return 0;
}

#ifdef FLOP
void triad_flop(real* A, real* B, real* C, int iter, int length)
{
    int i,j;
	
#pragma omp parallel for private(i,j)
    for(j = 0; j < iter; j++) 
    {
        for(i = 0; i < length; i++) 
        {
            A[i] = B[i] + 3.0 * C[i];
        }
    }
}
#endif

#ifdef BW
void triad_byte(real* A, real* B, int iter, int length)
{
    int i, j;
	
#pragma omp parallel for private(i,j)
    for(j = 0; j < iter; j++) 
    {
        for(i = 0; i < length; i++) 
        {
            A[i] = B[i];
        }
    }
}
#endif
