#include <stdio.h>
#define N 10

int main()
{
	int i, j, k;
	int a[N][N], b[N][N], c[N][N];

	for(i=0; i<N; i++) {
		for(j=0; j<i; j++) {
			a[i][j] = b[i][j] = c[i][j] = 0;
		}
		for(j=i; j<N; j++) {
			a[i][j] = i+j+1;
			b[i][j] = i+j+2;
			c[i][j] = 0;
		}
	}

	for(i=0; i<N; i++)
		for(j=i; j<N; j++)
			for(k=i; k<j+1; k++)
				c[i][j] += a[i][k] * b[k][j];

	for(i=0; i<N; i++) {
		for(j=0; j<N; j++)
			printf("%4d ", c[i][j]);
		printf("\n");
	}
}
