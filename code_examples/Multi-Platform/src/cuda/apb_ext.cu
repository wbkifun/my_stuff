#include "param1.h"
#include "param2.h"


__device__ double bpc(double b, double c) {
	return LLL*b + MM*c;
}
