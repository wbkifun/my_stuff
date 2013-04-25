#include <math.h>
#include <stdlib.h>

int main() {
	int i;
	float x;
	float *a, *b, *c;
	a = (float*)malloc(1000*sizeof(float));
	b = (float*)malloc(1000*sizeof(float));
	c = (float*)malloc(1000*sizeof(float));

	for(i=0; i<1000; i++) {
		x = -5+0.01*i;
		a[i] = exp(-x*x/2);
		b[i] = sin(5*x);
	}

	for(i=0; i<1000; i++) c[i] = a[i]*b[i];
}
