#include <stdio.h>

void add(int nx, int ny, void *ap, void *bp, void *cp) { 
	static int round = 1; 
	int i, j, idx;
	
	double *aa, *bb, *cc;
	aa = (double *)ap;
	bb = (double *)bp;
	cc = (double *)cp;
	printf("#C address: aa %p, bb %p, cc %p\n", aa, bb, cc);

	if ( round == 1)
		printf("*********************************************\n");

	for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
			idx = i*ny + j;
			cc[idx] = aa[idx] + bb[idx];
			printf("cc[%d,%d]= %g\n", i, j, cc[idx]);
		}
	}

	round += 1;
}
