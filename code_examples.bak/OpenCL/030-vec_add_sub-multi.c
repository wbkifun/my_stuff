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
	int i, tstep;
	int	num_gpu=3;
	size_t snx=256*1000000; // sub_nx, about 2G;
	size_t nx=num_gpu*snx;

	// Host allocation
	float *a_h;
	float *b_h;

	a_h = (float *)malloc(nx*sizeof(float));
	b_h = (float *)malloc(nx*sizeof(float));

	for(i=0; i<nx; i++) {
		a_h[i] = 1.0 + i*3e-6;
		b_h[i] = 2.0 + i*1e-6;
	}

	// Platform, Device, Context and Queue
	cl_int 				err;
	cl_platform_id		platform;
	cl_uint				num_devices;
	cl_device_id		devices[num_gpu];
	cl_context			context;
	cl_command_queue	queues[num_gpu];

	err = clGetPlatformIDs(1, &platform, NULL);
	CheckCL(err);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 3, devices, &num_devices);
	CheckCL(err);
	context = clCreateContext(0, num_gpu, devices, NULL, NULL, &err);
	CheckCL(err);
	for(i=0; i<num_gpu; i++) {
		queues[i] = clCreateCommandQueue(context, devices[i], 0, &err);
		CheckCL(err);
	}

	// Program and Kernel
	cl_program			program;
	cl_kernel			vecadd[num_gpu];
	cl_kernel			vecsub[num_gpu];
	char *CLFile = "vec_op.cl";
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
	for(i=0; i<num_gpu; i++) {
		vecadd[i] = clCreateKernel(program, "vecadd_gpu", &err);
		CheckCL(err);
		vecsub[i] = clCreateKernel(program, "vecsub_gpu", &err);
		CheckCL(err);
	}

	// Device buffer allocation
	cl_mem a_d[num_gpu];
	cl_mem b_d[num_gpu];
	for(i=0; i<num_gpu; i++) {
		a_d[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, snx*sizeof(cl_float), NULL, &err);
		CheckCL(err);
		b_d[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, snx*sizeof(cl_float), NULL, &err);
		CheckCL(err);

		err = clEnqueueWriteBuffer(queues[i], a_d[i], CL_FALSE, 0, snx*sizeof(cl_float), &a_h[i*snx], 0, NULL, NULL);
		CheckCL(err);
		err = clEnqueueWriteBuffer(queues[i], b_d[i], CL_FALSE, 0, snx*sizeof(cl_float), &b_h[i*snx], 0, NULL, NULL);
		CheckCL(err);
	}

	// Execute kernel
	for(i=0; i<num_gpu; i++) {
		err = clSetKernelArg(vecadd[i], 0, sizeof(cl_int), (void *)&nx);
		CheckCL(err);
		err = clSetKernelArg(vecadd[i], 1, sizeof(cl_mem), (void *)&a_d[i]);
		CheckCL(err);
		err = clSetKernelArg(vecadd[i], 2, sizeof(cl_mem), (void *)&b_d[i]);
		CheckCL(err);

		err = clSetKernelArg(vecsub[i], 0, sizeof(cl_int), (void *)&nx);
		CheckCL(err);
		err = clSetKernelArg(vecsub[i], 1, sizeof(cl_mem), (void *)&a_d[i]);
		CheckCL(err);
		err = clSetKernelArg(vecsub[i], 2, sizeof(cl_mem), (void *)&b_d[i]);
		CheckCL(err);
	}

	cl_event			evt_add[num_gpu];
	cl_event			evt_sub[num_gpu];
	cl_event			evt_read[num_gpu];
	for(i=0; i<num_gpu; i++) evt_sub[i] = CL_COMPLETE;
	size_t Ls=256;
	for(tstep=0; tstep<200; tstep++) {
		for(i=0; i<num_gpu; i++) {
			err = clEnqueueNDRangeKernel(queues[i], vecadd[i], 1, NULL, &nx, &Ls, 0, &evt_sub[i], &evt_add[i]);
			CheckCL(err);
			err = clEnqueueNDRangeKernel(queues[i], vecsub[i], 1, NULL, &nx, &Ls, 0, &evt_add[i], &evt_sub[i]);
			CheckCL(err);
		}
		//printf("tstep=%d\n", tstep);
	}

	//err = clEnqueueReadBuffer(queues[0], a_d[0], CL_TRUE, 0, snx*sizeof(cl_float), &b_h[0*snx], 0, 0, &evt_read[0]);
	//CheckCL(err);
	for(i=0; i<num_gpu; i++) {
		printf("i=%d\n", i);
		//err = clEnqueueReadBuffer(queues[i], a_d[i], CL_TRUE, 0, snx*sizeof(cl_float), &b_h[i*snx], 0, &evt_read[i-1], &evt_read[i]);
		err = clEnqueueReadBuffer(queues[i], a_d[i], CL_TRUE, 0, snx*sizeof(cl_float), &b_h[i*snx], 0, NULL, NULL);
		CheckCL(err);
	}

	printf("p1\n");
	// Verification
	float c0, c1;
	int sum=0;
	
	for(i=0; i<nx; i++) {
		//c0 = a_h[i] + b_h[i];
		//c0 = a_h[i] - b_h[i];
		c0 = a_h[i];
		c1 = b_h[i];

		if(c1 - c0 > 1e-2) {
			printf("[%d] %g - %g = %g\n", i, c1, c0, c1 - c0);
			sum += 1;
		}
	}
	if(sum == 0) printf("OK.\n");
	else printf("Result is wrong: %d times\n", sum);

	return 0;
}
