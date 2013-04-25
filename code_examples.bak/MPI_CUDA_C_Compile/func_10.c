#include "10.h"


void cfunc( int rank, float *data ) {
	int i;
	for ( i=0; i<CHUNKSIZE; i++ ) data[i] += rank;
}
