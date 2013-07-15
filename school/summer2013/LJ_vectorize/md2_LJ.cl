/*
  Molecular Dynamics - OpenCL Kernel 1
  Copyright, KISTI hsyi@kisti.re.kr
  Version: 2012. 06. 26
*/

#ifdef F32
#define scalar_t float
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define scalar_t double
#endif 

__kernel void computeAccel(const int N, __global scalar_t *r, __global scalar_t *a)
{
  
  int ip, jp;

  scalar_t x, y, z, rSqd, r2Inv, r6Inv, force;
  
  ip = get_global_id(0);

  if (ip >= N) return;
  
  a[3*ip+0] = 0;
  a[3*ip+1] = 0;
  a[3*ip+2] = 0;
  
  for ( jp = 0; jp < N; jp++)
    {
      if (ip == jp) continue;
    
      x = r[3*ip+0] - r[3*jp+0];
      y = r[3*ip+1] - r[3*jp+1];
      z = r[3*ip+2] - r[3*jp+2];
      
      rSqd = x*x + y*y + z*z;
      
      r2Inv = 1.0/rSqd;
      r6Inv = r2Inv * r2Inv * r2Inv;
      force = 24*r2Inv*r6Inv*(2*r6Inv - 1.0);
      
      a[3*ip+0] += x*force;
      a[3*ip+1] += y*force;
      a[3*ip+2] += z*force;
    }
}

__kernel void updateVelocities(const int N, const scalar_t dt, __global scalar_t *v, __global scalar_t *a)
{
  int ip;

  ip = get_global_id(0);

  if (ip >= N) return;
  
  v[3*ip +0] += 0.5 * a[3*ip +0] * dt;
  v[3*ip +1] += 0.5 * a[3*ip +1] * dt;
  v[3*ip +2] += 0.5 * a[3*ip +2] * dt;
}

__kernel void updateCoordinates(const int N, const scalar_t dt, __global scalar_t *r, __global scalar_t *v, __global scalar_t *a)
{
  int ip;

  ip = get_global_id(0);

  if (ip >= N) return;
  
  r[3*ip +0] += v[3*ip +0] * dt + 0.5 * a[3*ip +0] * dt * dt;
  r[3*ip +1] += v[3*ip +1] * dt + 0.5 * a[3*ip +1] * dt * dt;
  r[3*ip +2] += v[3*ip +2] * dt + 0.5 * a[3*ip +2] * dt * dt;
}
