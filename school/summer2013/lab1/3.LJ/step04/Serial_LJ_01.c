// KISTI educational molecular dynamics problem
// Copyright (c) 2012 - 2013 KISTI Supercomputing Center

// Lennard-Jones potential
// Periodic boundary condition

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifdef MKLRNG
#include <mkl.h>
#endif

int main()
{
  const int n     = 500;           // Number of atoms, molecules
  const int mt    = 20;         // Max time steps
  const int dtxyz = 100;           // Time interval to output xyz 

  int i;
  int j;

  double *x;
  double *v;
  double *f;

  const double domain = 300;       // Domain size (a.u.)
  const double dt     = 10;        // Time interval (a.u.)
  const double ms     = 0.0;       // Max speed (a.u.)
  const double em     = 1822.88839 * 28.0134; // Effective mass of N2
  const double lje    = 0.000313202;          // Lennard-Jones epsilon of N2
  const double ljs    = 6.908841465;          // Lennard-Jones sigma of N2

  #ifdef MKLRNG
  VSLStreamStatePtr stream; 

  vslNewStream(&stream, VSL_BRNG_MT19937,   5489); // Initiation, type, seed
  //vslNewStream(&stream, VSL_BRNG_SFMT19937, 5489); // Initiation, type, seed
  #endif
  /*
  x = (double *) malloc(n * 3 * sizeof(double));
  v = (double *) malloc(n * 3 * sizeof(double));
  f = (double *) malloc(n * 3 * sizeof(double));
  */

  x = (double *) _mm_malloc(n * 3 * sizeof(double),64);
  v = (double *) _mm_malloc(n * 3 * sizeof(double),64);
  f = (double *) _mm_malloc(n * 3 * sizeof(double),64);

  // Initialization


  for (i=0; i<n; i++)
    for (j=0; j<3; j++) x[i*3+j] = domain * rand() / (RAND_MAX + 1.0);

  for (i=0; i<n; i++)
    for (j=0; j<3; j++) v[i*3+j] = ms * (rand() / (RAND_MAX + 1.0) - 0.5);


  // Dynamics
  printf("#  Index    dTime    KinEng      PotEng       TotEng\n");
  for (i=0; i<mt; i++)
    {
      Force(n, lje, ljs, x, f);
      Solver(n, dt, em, x, v, f);

      Output_energy(i, n, dt, em, lje, ljs, x, v);
      if (i % dtxyz == 0) Output_xyz(i, n, x);
    }

  Output_xyz(i, n, x);

  _mm_free(x);
  _mm_free(v);
  _mm_free(f);
  
  return 0;
}

int Force(const int n, const double lje, const double ljs, const double *x, double *f)
{
  int j;
  int k;
  
  const double coe = -24 * lje;

  for (j=0; j<n; j++)
    {
      double d[3];

      double r, r2;

      f[j*3+0] = 0;
      f[j*3+1] = 0;
      f[j*3+2] = 0;

      for (k=0; k<n; k++)
        {
	  double PE;

          if (j == k) continue;

          d[0] = x[k*3+0] - x[j*3+0];
          d[1] = x[k*3+1] - x[j*3+1];
          d[2] = x[k*3+2] - x[j*3+2];

          r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
	  r2 = r * r;
	  
	  PE = 2 * pow(ljs / r, 12) - pow(ljs / r, 6);

          f[j*3+0] += PE * d[0] / r2;
          f[j*3+1] += PE * d[1] / r2;
          f[j*3+2] += PE * d[2] / r2;
        }

      f[j*3+0] *= coe;
      f[j*3+1] *= coe;
      f[j*3+2] *= coe;
    }

  return 0;
}

int Solver(const int n, const double dt, const double em, double *x, double *v, const double *f)
{
  int j, k;

  // Euler solver

  for (j=0; j<n; j++)
    {
      x[j*3+0] += v[j*3+0] * dt;
      x[j*3+1] += v[j*3+1] * dt;
      x[j*3+2] += v[j*3+2] * dt;

      v[j*3+0] += f[j*3+0] * dt / em;
      v[j*3+1] += f[j*3+1] * dt / em;
      v[j*3+2] += f[j*3+2] * dt / em;
    }

  return 0;
}

int Output_energy(const int i, const int n, const double dt, const double em, const double lje, const double ljs, const double *x, const double *v)
{
  int j, k;

  const double KEcoe = 0.5 * em;
  const double PEcoe = 0.5;

  double KE = 0, PE = 0, TE;

  FILE *fp;

  // Kinetic energy

  for (j=0; j<n; j++) KE += v[j*3+0] * v[j*3+0] + v[j*3+1] * v[j*3+1] + v[j*3+2] * v[j*3+2];
  KE *= KEcoe;
 
  // Potential energy

  for (j=0; j<n; j++) 
    {
      double d[3];

      double r;

      for (k=0; k<n; k++) 
	{
	  if (j == k) continue;

	  d[0] = x[k*3+0] - x[j*3+0];
	  d[1] = x[k*3+1] - x[j*3+1];
	  d[2] = x[k*3+2] - x[j*3+2];

	  r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

	  PE += 4 * lje * (pow(ljs / r, 12) - pow(ljs / r, 6));
	}
    }
  PE *= PEcoe;

  TE = KE + PE;
 
  // fp = stdout;
  if (i) fp = fopen("energy_orig.dat", "a");
  else   fp = fopen("energy_orig.dat", "w");

  if (fp == NULL) {printf(" Error: open a file.\n"); exit(1);}

  fprintf(fp, "%6d %10.5lf % lf % 1.10lf % 1.10lf\n", i, dt * i, KE, PE, TE);
  printf("%6d %10.5lf % lf % 1.10lf % 1.10lf\n", i, dt * i, KE, PE, TE);


  fclose(fp);

  return 0;
}

int Output_xyz(const int i, const int n, const double *x)
{
  int j;

  FILE *fp;

  if (i) fp = fopen("position.xyz", "a");
  else   fp = fopen("position.xyz", "w");

  if (fp == NULL) {printf(" Error: open a file.\n"); exit(1);}

  fprintf(fp, "%d\n\n", n);
  for (j=0; j<n; j++) fprintf(fp, "N % lf % lf % lf\n", x[j*3+0], x[j*3+1], x[j*3+2]);
  //fprintf(fp, "\n");

  fclose(fp);

  return 0;
}
