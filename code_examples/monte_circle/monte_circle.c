#include <stdio.h>
#include <stdlib.h>
#include <time.h>	// 코드수행시간을 측정하는 함수호출


int main(void)
{
	int N,n,i; 
	clock_t start_time, end_time;	// 시작시간과끝나는시간 
	double duration_time; 
	double Pi,x,y; 
	
	n=0; 
	printf("전체점을 찍을 횟수 N = "); 
	srand(12356);	//임의의시작점 
	scanf("%d", &N);
	start_time = clock(); // start time 
	
	for (i=0; i<N; i++) 
	{ 
		x= (double) rand()/RAND_MAX ; 
		y= (double) rand()/RAND_MAX ; 
		
		if (x*x+y*y<=1) n++; 
	} 
	
	end_time = clock(); // end time
	Pi= (double) 4* n / N ; 
	duration_time = (double)(end_time-start_time)/CLOCKS_PER_SEC; 
	printf("시행횟수 %d 번일 때 Pi= %f \n", N, Pi); 
	printf("수행시간은 %f 입니다.\n", duration_time); 
	
	return 0; 
}
