#include <stdio.h>

void cfunc(int rows, int cols, void *p) { 
	static int round = 1; 
	int i, j, idx;
	
	double *dptr;
	dptr = (double *)p;
	printf("#C address: %p\n", dptr);

	if ( round == 1 || round == 3 )
		printf("*********************************************\n");
	printf(" round %d\n", round);

	for (i=0; i<rows; i++) {
		for (j=0; j<cols; j++) {
			idx = i*cols + j;
			printf("[%d] arr[%d,%d]= %g\n", round, i, j, dptr[idx]);
		}
	}

	round += 1;
}
