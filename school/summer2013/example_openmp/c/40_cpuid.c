#include <stdio.h>
#include <sched.h>

int main()
{
#pragma omp parallel
{
	printf("cpuid : %d\n", sched_getcpu() );
}
}
