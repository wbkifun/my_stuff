#include <stdio.h>
#include <stdlib.h>

void print_array(int n, char str, float *a) {
	int i;
	printf("%c:  ", str);
	for(i=0; i<n; i++) printf("\t%f", a[i]);
	printf("\n");
}

void vecadd(int n, float *a, float *b, float *c) {
	int i;
	for(i=0; i<n; i++) {
		c[i] = a[i] + b[i];
	}
}


int main() {
	int i;
	int n=5;
	float *a, *b, *c;

	// allocation in host memory
	a = (float *)malloc(n*sizeof(float));
	b = (float *)malloc(n*sizeof(float));
	c = (float *)malloc(n*sizeof(float));

	// initialize
	for(i=0; i<n; i++) {
		//a[i] = i+5.5;
		//b[i] = -1.2*i;
		a[i] = rand()/(RAND_MAX+1.);
		b[i] = rand()/(RAND_MAX+1.);
	}
	print_array(n, 'a', a);
	print_array(n, 'b', b);

	// function call
	vecadd(n, a, b, c);
	printf("results from CPU\n");
	print_array(n, 'c', c);

	free(a);
	free(b);
	free(c);
	return 0;
}
