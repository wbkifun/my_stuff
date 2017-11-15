#include <math.h>


void my_cos(int nx, double *in_arr, double *out_arr) {
	int i;

	for(i = 0; i < nx; i++) {
		out_arr[i] = cos(in_arr[i]);
	}
}



void my_cos_2d(int nx, int ny, double *in_arr, double *out_arr) {
	int i, j;

	for(i = 0; i < nx; i++) {
		for(j = 0; j < ny; j++) {
			out_arr[i*ny+j] = cos(in_arr[i*ny+j]);
		}
	}
}



void my_cos_2d_2ptr(int nx, int ny, double **in_arr, double **out_arr) {
	int i, j;

	for(i = 0; i < nx; i++) {
		for(j = 0; j < ny; j++) {
			out_arr[i][j] = cos(in_arr[i][j]);
		}
	}
}
