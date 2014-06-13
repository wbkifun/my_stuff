/*
  Molecular Dynamics - OpenCL Kernel 1
  Copyright, KISTI hsyi@kisti.re.kr
  Version: 2012. 06. 26

  1. Normal CPU version
  2. OpenCL accelerator version
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#ifdef CL
#include <CL/cl.h>
#endif

#ifdef F32
#define scalar_t float
#else
#define scalar_t double
#endif 

int initialize(int N, scalar_t* pos, scalar_t* vel, scalar_t* acc);

#ifdef CL
int velocityVerlet_OnAcc(int N, scalar_t dt, scalar_t* pos, scalar_t* vel, scalar_t* acc, int pid, int did);
#else
int velocityVerlet_OnCPU(int N, scalar_t dt, scalar_t* pos, scalar_t* vel, scalar_t* acc);
#endif

scalar_t instantaneousTemperature(int N, scalar_t *vel) ;
int computeAccelerations_OnGPU(int N, scalar_t* pos, scalar_t* acc);

int main()
{
  const int N = 25600;
  const int PID = 0;
  const int DID = 0;

  int i; 

  scalar_t  *pos = (scalar_t*) malloc( 3 * N * sizeof(scalar_t)); 
  scalar_t  *vel = (scalar_t*) malloc( 3 * N * sizeof(scalar_t)); 
  scalar_t  *acc = (scalar_t*) malloc( 3 * N * sizeof(scalar_t));

  const scalar_t dt = 0.01;

  double eTime;

  struct timespec Tstart, Tend;
 
  initialize(N, pos, vel, acc);

  clock_gettime(CLOCK_REALTIME, &Tstart);
#ifdef CL
  printf("Compiled with OpenCL \n\n");
  
  velocityVerlet_OnAcc(N, dt, pos, vel, acc, PID, DID);

#else
  printf("Compiled on CPU.. \n\n");

  velocityVerlet_OnCPU(N, dt, pos, vel, acc);
#endif
  clock_gettime(CLOCK_REALTIME, &Tend);
  eTime = (Tend.tv_sec-Tstart.tv_sec)+(Tend.tv_nsec-Tstart.tv_nsec) * 1e-9;

  printf("CPU/Acc time=%8.4lf (sec)\n",eTime);
  
  free(pos);
  free(vel);
  free(acc);
}

int initialize(int N, scalar_t* pos, scalar_t* vel, scalar_t* acc)
{
  int ix, iy, iz, ip, k; 
  int nsize;

  scalar_t Domain = 100; 
  scalar_t vMax = 0.1;  
  scalar_t region;
  
  nsize = (int)(ceil(pow((scalar_t)(N), 1.0/3)));
  region = Domain / nsize;

  ip = 0;
  for ( ix = 0; ix < nsize; ix++)
    for ( iy = 0; iy < nsize; iy++)
      for ( iz = 0; iz < nsize; iz++)
	{
	  if (ip < N)
	    {
	      pos[3*ip + 0] = (ix + 0.5) * region;
	      pos[3*ip + 1] = (iy + 0.5) * region;
	      pos[3*ip + 2] = (iz + 0.5) * region;
	    }
	  ip ++;
	}
  
  srand48(100);
  
  for ( ip = 0; ip < N; ip++)
    {
      vel[3*ip + 0] =  vMax * (2*drand48() -1);
      vel[3*ip + 1] =  vMax * (2*drand48() -1);
      vel[3*ip + 2] =  vMax * (2*drand48() -1);
    }

  if (1)
    {
      int i;
      
      FILE *fp;
      
      fp = fopen("position.0", "w");
      if (fp == NULL) {printf(" Error: write position.0\n"); exit(1);}
      
      for (i=0; i<N; i++)
        fprintf(fp, "%5d %8.4lf %8.4lf %8.4lf %8.4lf %8.4lf %8.4lf \n", i, pos[3*i+0], pos[3*i+1], pos[3*i+2], vel[3*i+0], vel[3*i+1], vel[3*i+2]);
      
      fclose(fp);
    }
    return 0;
}

int computeAccelerations_OnCPU(int N, scalar_t* pos, scalar_t* acc)
{
    int ip, jp;

  scalar_t x, y, z, rSqd, r2Inv, r6Inv, force;

  #pragma omp parallel for private(jp, x, y, z, rSqd, r2Inv, r6Inv, force)
  for ( ip = 0; ip < N; ip++)
    {
      acc[3*ip+0] = 0;
      acc[3*ip+1] = 0;
      acc[3*ip+2] = 0;
      
      for ( jp = 0; jp < N; jp++) 
	{
	  if (ip == jp) continue;

	  x = pos[3*ip+0] - pos[3*jp+0];
	  y = pos[3*ip+1] - pos[3*jp+1];
	  z = pos[3*ip+2] - pos[3*jp+2];
	  
	  rSqd = x*x + y*y + z*z;
	  
	  r2Inv = 1.0/rSqd;
	  r6Inv = r2Inv * r2Inv * r2Inv;
	  force = 24*r2Inv*r6Inv*(2*r6Inv - 1.0);
      
	  acc[3*ip+0] += x*force;
	  acc[3*ip+1] += y*force;
	  acc[3*ip+2] += z*force;
	}
    }

  return 0;
}

int velocityVerlet_OnCPU(int N, scalar_t dt, scalar_t* pos, scalar_t* vel, scalar_t* acc)
{
  int tstep, ip;

  scalar_t temp; 

  for ( tstep = 0; tstep < 3; tstep++)
    {
      computeAccelerations_OnCPU(N, pos, acc);
    
      #pragma omp parallel for 
      for ( ip = 0; ip < N; ip++)
	{
	  vel[3*ip + 0] += 0.5 * acc[3*ip + 0] * dt;
	  vel[3*ip + 1] += 0.5 * acc[3*ip + 1] * dt;
	  vel[3*ip + 2] += 0.5 * acc[3*ip + 2] * dt;
	}
      
      #pragma omp parallel for
      for ( ip = 0; ip < N; ip++)
	{
	  pos[3*ip + 0] += vel[3*ip + 0 ] * dt + 0.5*acc[3*ip + 0]*dt*dt;
	  pos[3*ip + 1] += vel[3*ip + 1 ] * dt + 0.5*acc[3*ip + 1]*dt*dt;
	  pos[3*ip + 2] += vel[3*ip + 2 ] * dt + 0.5*acc[3*ip + 2]*dt*dt;
	}
      
    computeAccelerations_OnCPU(N, pos, acc);

    #pragma omp parallel for
    for ( ip = 0; ip < N; ip++)
      {
	vel[3*ip + 0] += 0.5 * acc[3*ip + 0 ] * dt;
	vel[3*ip + 1] += 0.5 * acc[3*ip + 1 ] * dt;
	vel[3*ip + 2] += 0.5 * acc[3*ip + 2 ] * dt;
      }
    }
  
  temp = instantaneousTemperature(N, vel);
  
  printf("\nTemperature=%8.6lf \n", temp);

  if (1)
    {
      int i;

      FILE *fp;
      
      fp = fopen("position.CPU", "w");
      if (fp == NULL) {printf(" Error: write position.CPU\n"); exit(1);}
      
      for (i=0; i<N; i++)
	fprintf(fp, "%5d %8.4lf %8.4lf %8.4lf %8.4lf %8.4lf %8.4lf \n", i, pos[3*i+0], pos[3*i+1], pos[3*i+2], vel[3*i+0], vel[3*i+1], vel[3*i+2]);
      
      fclose(fp);
    }

  return 0;
}

scalar_t instantaneousTemperature(int N, scalar_t *vel)
{
  int ip;  
  
  scalar_t x,y,z,sum;
  
  x = y = z = 0;
  #pragma omp parallel for private(x, y, z)
  for ( ip = 0; ip < N; ip++)
    {
      x += vel[3*ip+0]*vel[3*ip+0];
      y += vel[3*ip+1]*vel[3*ip+1];
      z += vel[3*ip+2]*vel[3*ip+2];
    }
 
  sum = (x+y+z)/(3.0 * (N - 1));
  
  return sum ;
}

#ifdef CL
int velocityVerlet_OnAcc(int N, scalar_t dt, scalar_t* pos, scalar_t* vel, scalar_t* acc, int PID, int DID)
{
  int Ret;
  int k;
  int tstep, index;
  int i;

  const size_t __SPN   = N * 3 * sizeof(scalar_t);  
  size_t           GWS = N;        // Global work size
  size_t           LWS = 32;         // Local  work size

  scalar_t temp;

  double elapsedTime =0;

#ifdef F32
  char *CL_OPT   = "-DF32";     // Flags to OpenCL compiler
#else  //  F64
  char *CL_OPT   = NULL;
#endif

  FILE *fp;
  
  cl_platform_id   cpPlatform[3];   
  cl_device_id     cdDevice;
  cl_context       cxContext;
  cl_program       cpProgram;
  cl_kernel        ckKernel[10];
  cl_command_queue cqCommandQueue;
  cl_mem cmDevPOS;
  cl_mem cmDevVEL;
  cl_mem cmDevACC;
  cl_mem_flags flag1 = CL_MEM_READ_WRITE;
  cl_mem_flags flag2 = CL_MEM_WRITE_ONLY;
  
  printf("\nGlobal Work Size= %zu \nLocal Work Size= %zu\n\n", GWS, LWS);
  
  CL_Setup_PDC(cpPlatform, &cdDevice, &cxContext, PID, DID);
  CL_Setup_PK(cdDevice, cxContext, &cpProgram, ckKernel, CL_OPT);
  
  cqCommandQueue = clCreateCommandQueue(cxContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &Ret); CheckFN(Ret);
  
  // OpenCL BUFFER 
  
  cmDevPOS    = clCreateBuffer(cxContext, flag1,  __SPN, NULL, &Ret); CheckFN(Ret);
  cmDevVEL    = clCreateBuffer(cxContext, flag1,  __SPN, NULL, &Ret); CheckFN(Ret);
  cmDevACC    = clCreateBuffer(cxContext, flag2,  __SPN, NULL, &Ret); CheckFN(Ret);
  
  k = 0;
  Ret = clSetKernelArg(ckKernel[0], k++, sizeof(cl_int),(void*)&N);
  Ret = clSetKernelArg(ckKernel[0], k++, sizeof(cl_mem),(void*)&cmDevPOS);
  Ret = clSetKernelArg(ckKernel[0], k++, sizeof(cl_mem),(void*)&cmDevACC);
  
  /* updateVelocities */
  k = 0; 
  Ret = clSetKernelArg(ckKernel[1], k++, sizeof(cl_int),(void*)&N);
  Ret = clSetKernelArg(ckKernel[1], k++, sizeof(scalar_t),(void*)&dt);
  Ret = clSetKernelArg(ckKernel[1], k++, sizeof(cl_mem),(void*)&cmDevVEL);
  Ret = clSetKernelArg(ckKernel[1], k++, sizeof(cl_mem),(void*)&cmDevACC);
  
  /* updateCoordinate */
  k = 0; 
  Ret = clSetKernelArg(ckKernel[2], k++, sizeof(cl_int),(void*)&N);
  Ret = clSetKernelArg(ckKernel[2], k++, sizeof(scalar_t),(void*)&dt);
  Ret = clSetKernelArg(ckKernel[2], k++, sizeof(cl_mem),(void*)&cmDevPOS);
  Ret = clSetKernelArg(ckKernel[2], k++, sizeof(cl_mem),(void*)&cmDevVEL);
  Ret = clSetKernelArg(ckKernel[2], k++, sizeof(cl_mem),(void*)&cmDevACC);
  
  Ret = clEnqueueWriteBuffer(cqCommandQueue, cmDevPOS, CL_FALSE, 0, __SPN, (void *)pos, 0, NULL, NULL); CheckFN(Ret);
  Ret = clEnqueueWriteBuffer(cqCommandQueue, cmDevVEL, CL_FALSE, 0, __SPN, (void *)vel, 0, NULL, NULL); CheckFN(Ret);

  for ( tstep = 0; tstep < 3; tstep++)
    {
      /* computeAccel */
      Ret = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel[0], 1, NULL, &GWS, &LWS, 0, NULL, NULL);CheckFN(Ret);
      
      /* updateVelocities */
      Ret = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel[1], 1, NULL, &GWS, &LWS, 0, NULL, NULL);CheckFN(Ret);
      
      /* updateCoordinates */
      Ret = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel[2], 1, NULL, &GWS, &LWS, 0, NULL, NULL);CheckFN(Ret);
      
      /* computeAccel */
      Ret = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel[0], 1, NULL, &GWS, &LWS, 0, NULL, NULL);CheckFN(Ret);

      /* updateVelocities */
      Ret = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel[1], 1, NULL, &GWS, &LWS, 0, NULL, NULL);CheckFN(Ret);
    }
  
  Ret = clEnqueueReadBuffer(cqCommandQueue, cmDevVEL, CL_FALSE, 0, __SPN, (void *) vel, 0, NULL, NULL); CheckFN(Ret);
  Ret = clEnqueueReadBuffer(cqCommandQueue, cmDevPOS, CL_TRUE,  0, __SPN, (void *) pos, 0, NULL, NULL); CheckFN(Ret);
  
  temp = instantaneousTemperature(N, vel);

  printf("\nTemperature=%8.6lf\n",temp);

  fp = fopen("position.CL", "w");
  if (fp == NULL) {printf(" Error: write position.CL\n"); exit(1);}
  
  for (i=0; i<N; i++)
    fprintf(fp, "%5d %8.4lf %8.4lf %8.4lf %8.4lf %8.4lf %8.4lf \n", i, pos[3*i+0], pos[3*i+1], pos[3*i+2], vel[3*i+0], vel[3*i+1], vel[3*i+2]);
  
  fclose(fp);
  
  clReleaseMemObject(cmDevPOS);
  clReleaseMemObject(cmDevVEL);
  clReleaseMemObject(cmDevACC);
  
  clReleaseProgram(cpProgram);
  clReleaseCommandQueue(cqCommandQueue);
  clReleaseContext(cxContext);

  return 0;
}

int CL_Setup_PDC(cl_platform_id *cpPlatform, cl_device_id *cdDevice, cl_context *cxContext, const int PID, const int DID)
{
  // [Description] OpenCL Setup; Stage 1; 1 platform, 1 device, 1 context
	
  int i;
  int devIdx;
  int numDevices; 

  size_t retsize_t;
  size_t longsize;

  char InfoString[1000];         
  char buftxt[1024];

  cl_int ret;

  cl_uint NumPD = 1;

  printf("[OpenCL Setup : PDC]\n");

  ret = clGetPlatformIDs(0, NULL, &NumPD); CheckFN(ret);
  printf("CL_PLATFORM_NUM    : %u\n", NumPD);
  ret = clGetPlatformIDs(NumPD, cpPlatform, NULL); CheckFN(ret);
  
  for (i=0; i<NumPD; i++)
    {
      ret = clGetPlatformInfo(cpPlatform[i], CL_PLATFORM_NAME, 0, NULL, &retsize_t); CheckFN(ret);
      ret = clGetPlatformInfo(cpPlatform[i], CL_PLATFORM_NAME, retsize_t, (void *) InfoString, NULL); CheckFN(ret); 
      printf("CL_PLATFORM_NAME %d : %s\n", i, InfoString);
    }
  printf("----------------------------------------------------------------------------------\n\n");
  
  ret = clGetPlatformInfo(cpPlatform[PID], CL_PLATFORM_NAME, 0, NULL, &retsize_t); CheckFN(ret);
  ret = clGetPlatformInfo(cpPlatform[PID], CL_PLATFORM_NAME, retsize_t, (void *) InfoString, NULL);
  printf("CHOSEN_PLATFORM  %d : %s\n", PID,InfoString);
  
  ret = clGetDeviceIDs(cpPlatform[PID], CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numDevices); CheckFN(ret);
  ret = clGetDeviceIDs(cpPlatform[PID], CL_DEVICE_TYPE_ACCELERATOR, numDevices, cdDevice, 0); CheckFN(ret);
  
  *cxContext = clCreateContext(0, numDevices, cdDevice, NULL, NULL, &ret); CheckFN(ret);
  



  ret = clGetDeviceInfo(*cdDevice, CL_DEVICE_NAME, 0, NULL, &retsize_t); CheckFN(ret);
  ret = clGetDeviceInfo(*cdDevice, CL_DEVICE_NAME, retsize_t, (void *) InfoString, NULL); CheckFN(ret);
	
  clGetDeviceInfo(cdDevice[DID],CL_DEVICE_NAME, sizeof(buftxt), buftxt,NULL);
  printf("Device #%d name = %s\n", DID, buftxt);

  clGetDeviceInfo(cdDevice[DID],CL_DRIVER_VERSION, sizeof(buftxt), buftxt,NULL);
  printf("\tDriver version      : %s\n", buftxt);
	
  clGetDeviceInfo(cdDevice[DID],CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(longsize),&longsize,NULL);
  printf("\tGlobal Memory       : %zu (MB) \n",longsize/1024/1024);
	
  clGetDeviceInfo(cdDevice[DID],CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(longsize),&longsize,NULL);
  printf("\tGlobal Memory Cache : %lu (MB) \n",longsize/1024/1024);
	
  clGetDeviceInfo(cdDevice[DID],CL_DEVICE_LOCAL_MEM_SIZE, sizeof(longsize),&longsize,NULL);
  printf("\tLocal Memory        : %zu (KB)\n",longsize/1024);
	
  clGetDeviceInfo(cdDevice[DID],CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(longsize),&longsize,NULL);
  printf("\tMax clock           : %lu (MHz) \n",longsize);
	
  clGetDeviceInfo(cdDevice[DID],CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(longsize),&longsize,NULL);
  printf("\tMAX_WORK_GROUP_SIZE : %lu \n",longsize);
	
  clGetDeviceInfo(cdDevice[DID],CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(longsize),&longsize,NULL);
  printf("\tMAX_COMPUTE_UNITS   : %lu \n",longsize);
  
  printf("-----------------------------------------------------------------------\n");
	
  return 0;
}

int CL_Setup_PK(const cl_device_id cdDevice, const cl_context cxContext,
		cl_program *cpProgram, cl_kernel *ckKernel, const char *CL_OPT)
{
  // [Description] OpenCL Setup; Stage 2; 1 program and kernels

  const int NumKernel = 3;

  int i;
	
  size_t SourceLength;
  size_t retsize_t;
  
  const char *CLFile     = "md2_LJ.cl"; // CL source filename
  const char *CLKernel[] = {"computeAccel", "updateVelocities",  "updateCoordinates"};
	
  char *cSourceCL;
  char  InfoString[30000];     
  
  FILE *fp;
  
  cl_int Ret;

  printf("\n[OpenCL Setup : PK]\n");
  
  fp = fopen(CLFile, "r");
  if (fp == NULL) {printf("%s is needed.\n", CLFile); exit(1);}
  
  fseek(fp, 0, SEEK_END); 
  SourceLength = ftell(fp);
  fseek(fp, 0, SEEK_SET); 
  
  cSourceCL = (char *) malloc((SourceLength + 1) * sizeof(char)); 
  
  fread(cSourceCL, SourceLength, 1, fp);
  fclose(fp);
  
  cSourceCL[SourceLength] = '\0';
  
  *cpProgram = clCreateProgramWithSource(cxContext, 1, (const char **)&cSourceCL, &SourceLength, &Ret); CheckFN(Ret);
  Ret = clBuildProgram(*cpProgram, 0, NULL, CL_OPT, NULL, NULL);
  
  if (Ret != CL_SUCCESS)  
    {
      printf("Error in building kernels.\n");
      clGetProgramBuildInfo(*cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &retsize_t);
      clGetProgramBuildInfo(*cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, retsize_t, (void *) InfoString, NULL);
      printf("%s\n", InfoString);
      exit(1);
    }
  
  for (i=0; i<NumKernel; i++) ckKernel[i] = clCreateKernel(*cpProgram, CLKernel[i], &Ret); CheckFN(Ret);
   
  return 0;
}

int CheckFN(const cl_int err)
{
  if (err != CL_SUCCESS)
    {
      printf("CL_FAILED: %d\n", err);
      exit(1);
    }

  return 0;
}
#endif
