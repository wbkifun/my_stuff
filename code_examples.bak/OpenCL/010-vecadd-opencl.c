#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>


CheckCL(cl_int err) {
	if(err != CL_SUCCESS) {
		printf("CL_FAILED: %d\n", err);
		exit(1);
	}
}


int main() {
	int i;
	size_t nx=256*80000; //512;

	// Host allocation
	float *a_h;
	float *b_h;
	float *c_h;

	a_h = (float *)malloc(nx*sizeof(float));
	b_h = (float *)malloc(nx*sizeof(float));
	c_h = (float *)malloc(nx*sizeof(float));

	for(i=0; i<nx; i++) {
		a_h[i] = 1.0 + i*3e-6;
		b_h[i] = 2.0 + i*1e-6;
	}

	// Platform, Device, Context and Queue
	cl_int 				err;
	cl_platform_id		platform;
	cl_device_id		device;
	cl_context			context;
	cl_command_queue	queue;

	err = clGetPlatformIDs(1, &platform, NULL);
	CheckCL(err);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CheckCL(err);
	context = clCreateContext(0, 1, &device, NULL, NULL, &err);
	CheckCL(err);
	queue = clCreateCommandQueue(context, device, 0, &err);
	CheckCL(err);

	// Program and Kernel
	cl_program			program;
	cl_kernel			kernel;
	cl_event			event;
	char *CLFile = "vec_add.cl";
	FILE *fp;
	size_t SourceLength;
	char *cSourceCL;
	
	fp = fopen(CLFile, "rb");
	if(fp == NULL) {
		printf("%s is needed.\n", CLFile); 
		exit(1);
	}
	fseek(fp, 0, SEEK_END);
	SourceLength = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	cSourceCL = (char *)malloc((SourceLength+1)*sizeof(char));
	fread(cSourceCL, SourceLength, 1, fp);
	fclose(fp);
	cSourceCL[SourceLength] = '\0';

	program = clCreateProgramWithSource(context, 1, (const char **)&cSourceCL, &SourceLength, &err);
	CheckCL(err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CheckCL(err);
	kernel = clCreateKernel(program, "vecadd_gpu", &err);
	CheckCL(err);

	// Device buffer allocation
	cl_mem a_d;
	cl_mem b_d;
	cl_mem c_d;
	a_d = clCreateBuffer(context, CL_MEM_READ_ONLY, nx*sizeof(cl_float), NULL, &err);
	CheckCL(err);
	b_d = clCreateBuffer(context, CL_MEM_READ_ONLY, nx*sizeof(cl_float), NULL, &err);
	CheckCL(err);
	c_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nx*sizeof(cl_float), NULL, &err);
	CheckCL(err);

	err = clEnqueueWriteBuffer(queue, a_d, CL_FALSE, 0, nx*sizeof(cl_float), a_h, 0, NULL, NULL);
	CheckCL(err);
	err = clEnqueueWriteBuffer(queue, b_d, CL_FALSE, 0, nx*sizeof(cl_float), b_h, 0, NULL, NULL);
	CheckCL(err);

	// Execute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&nx);
	CheckCL(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&a_d);
	CheckCL(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_d);
	CheckCL(err);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&c_d);
	CheckCL(err);

	size_t Ls=256;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &nx, &Ls, 0, NULL, &event);
	CheckCL(err);

	err = clEnqueueReadBuffer(queue, c_d, CL_TRUE, 0, nx*sizeof(cl_float), c_h, 0, NULL, NULL);
	CheckCL(err);

	// Verification
	float c0, c1;
	int sum=0;
	c0 = a_h[i] + b_h[i];
	c1 = c_h[i];
	
	for(i=0; i<nx; i++) {
		if(c1 - c0 > 1e-6) {
			printf("[%d] %g - %g = %g\n", i, c1, c0, c1 - c0);
			sum += 1;
		}
	}
	if(sum == 0) printf("OK.\n");
	else printf("Result is wrong: %d times\n", sum);

	return 0;
}
