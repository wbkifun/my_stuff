#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel num_threads(32)
{
	printf("A ");
	printf("B ");
	printf("C ");
	printf("D ");
	printf("\n");
} // pragma omp parallel
}
