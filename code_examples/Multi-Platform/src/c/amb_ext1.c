#include "param2.h"
#include "amb_ext2.h"


void bmc(int nx, double ll, double *b, double *c) {
	// size and intent of array arguments for f2py
	// b :: nx, in
	// c :: nx, inout
	
	int i;

	for (i=0; i<nx; i++) {
		c[i] = ll*b[i] + mc(MM, c[i]);
	}
}
