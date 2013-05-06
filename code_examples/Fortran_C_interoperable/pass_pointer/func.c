#include <stdio.h>

void cfunc(int rows, int cols, void *p) { 
	static int round = 1; 
	
	double *dptr;
	dptr=(double *)p;
	printf("#C address: %p\n", dptr);
	int i;

	if ( 1 == round ) {
		printf("*********************************************\n");
		printf(" [<call#>] <var>[<C idx>] (<F idx>) = <value>\n");
		printf("*********************************************\n");
	}
	for (i=0; i<rows; i++)
		printf(" [%d] ptr[%3d,  0] (  1,%3d) = %f\n",round,i,i+1,dptr[i*cols]);

	for (i=0; i<cols; i++)
		printf(" [%d] ptr[  0,%3d] (%3d,  1) = %f\n",round,i,i+1,dptr[i]);

	round=round+1;
	return;
}
