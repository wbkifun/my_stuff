// This example from an alpha release of the KISTI SummerSchool 2013
// Contact: Hongsuk Yi <hsyi@kisti.re.kr>
// Copyright (c) 2013, KISTI
// All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

double mysecond()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return (double)(t.tv_sec + t.tv_usec*1e-6);
}

#define REAL        float
#define NITER       1000
#define SIZE        1024
int main(int argc, char **argv)
{
	int i,j,k;

    int BytesPerWord;
    BytesPerWord = sizeof(REAL);
    printf("Uses %d bytes per word.\n", BytesPerWord);
	printf("Array Size = %d \n",SIZE);
    printf("Total memory required = %.3lf MB.\n",
		   (2.0 * SIZE * BytesPerWord) * 1e-6);
	
	double sTime, eTime;
	double scalar = 3.0;

	REAL *a, *b;

	//	a = (REAL*) malloc(sizeof(REAL)*SIZE);
	//	b = (REAL*) malloc(sizeof(REAL)*SIZE);
	a = (REAL*) _mm_malloc(sizeof(REAL)*SIZE,64);
	b = (REAL*) _mm_malloc(sizeof(REAL)*SIZE,64);
		
    for (i=0; i<SIZE; i++) {
		a[i] = 1.0;
		b[i] = 2.0;
	}

	sTime = mysecond();
	for (i =0; i< NITER; i++)
	{
		for (j=0; j< SIZE; j++)
		{
			a[j] = b[j] + scalar * a[j];
		}
	}
	eTime = mysecond() - sTime;
	double gflops;
	gflops = (1.0e-9*NITER*SIZE*2)/eTime;
	printf("SIZE=%9d \nETime=%8.4lf(sec) \nGflops=%8.3lf(Gflop/s)\n",
			   SIZE, eTime, gflops);

	_mm_free(a);
	_mm_free(b);
}


