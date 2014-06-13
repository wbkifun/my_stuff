// This example from an alpha release of the KISTI SummerSchool 2013
// Intel MIC architecture
// Contact: Hongsuk Yi <hsyi@kisti.re.kr>
// Copyright (c) 2013, KISTI
// All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <omp.h>

double mysecond()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return (double)(t.tv_sec + t.tv_usec*1e-6);
}

#define REAL             float
#define NITER            100000000
#define SIZE             (1024*1024)
#define LOOP             64


int main(int argc, char **argv)
{
	int i,j,k;

    int BytesPerWord;
    BytesPerWord = sizeof(REAL);
    printf("Uses %d bytes per word.\n", BytesPerWord);
	printf("Array Size = %d \n",SIZE);
    printf("Total memory required = %.3lf MB.\n",
		   (3.0 * SIZE * BytesPerWord) * 1e-6);
	
	double sTime, eTime;
	double scalar = 3.0;

	//	omp_set_num_threads(2);
	int nthreads,tid;

#pragma omp parallel
#pragma omp master
    nthreads = omp_get_num_threads();
	
	//#pragma omp parallel private(tid)    {
	//		nthreads = omp_get_num_threads();
	//		tid      = omp_get_thread_num();
	//    }
	printf ("nthreads=%d\n",nthreads);
	
	REAL *a, *b;

	a = (REAL*) _mm_malloc(sizeof(REAL)*SIZE,64);
	b = (REAL*) _mm_malloc(sizeof(REAL)*SIZE,64);
	
#pragma omp parallel for	
    for (i=0; i<SIZE; i++) {
		a[i] = 1.0;
		b[i] = 2.0;
	}

	int ioff;
	
	sTime = mysecond();
#pragma omp parallel for private(j,k)
	for (i =0; i < nthreads; i++)
	{
		ioff = i * LOOP;
		for (j =0; j< NITER; j++)
		{
			for (k=0; k< LOOP; k++)
			{
				a[k+ioff] = a[k+ioff] + scalar * b[k+ioff];
			}
		}
	}
	eTime = mysecond() - sTime;
	double gflops;
	gflops = (1.0e-9*nthreads*NITER*LOOP*2)/eTime;
	printf("SIZE=%9d \t ETime=%8.4lf \t Gflops=%8.3lf\n",
		   SIZE, eTime, gflops);

	_mm_free(a);
	_mm_free(b);
}


