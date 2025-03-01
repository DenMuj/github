/**
 * DBEC-GP-OMP-CUDA-MPI programs are developed by:
 *
 * Vladimir Loncar, Antun Balaz
 * (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
 *
 * Srdjan Skrbic
 * (Department of Mathematics and Informatics, Faculty of Sciences, University of Novi Sad, Serbia)
 *
 * Paulsamy Muruganandam
 * (Bharathidasan University, Tamil Nadu, India)
 *
 * Luis E. Young-S, Sadhan K. Adhikari
 * (UNESP - Sao Paulo State University, Brazil)
 *
 *
 * Public use and modification of these codes are allowed provided that the
 * following papers are cited:
 * [1] V. Loncar et al., Comput. Phys. Commun. 209 (2016) 190.
 * [2] V. Loncar et al., Comput. Phys. Commun. 200 (2016) 406.
 * [3] R. Kishor Kumar et al., Comput. Phys. Commun. 195 (2015) 117.
 *
 * The authors would be grateful for all information and/or comments
 * regarding the use of the programs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <complex.h>
#include <omp.h>
#include <sys/time.h>
#include <mpi.h>

#define MAX(a, b, c)       (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c)
#define MAX_FILENAME_SIZE  256
#define RMS_ARRAY_SIZE     4
#define EPSILON            1e-20

#define BOHR_RADIUS        5.2917720859e-11

#define FFT_FLAG FFTW_MEASURE
/* #define FFT_FLAG FFTW_ESTIMATE */

static enum OutputFlags { DEN_X00 = 1 << 0, DEN_X = 1 << 1, DEN_Y = 1 << 2, DEN_Z = 1 << 3, DEN_XY = 1 << 4, DEN_XZ = 1 << 5, DEN_YZ = 1 << 6, DEN_XY0 = 1 << 7, DEN_X0Z = 1 << 8, DEN_0YZ = 1 << 9, DEN_XYZ = 1 << 10, ARG_XY0 = 1 << 11, ARG_X0Z = 1 << 12, ARG_0YZ = 1 << 13 } outflags;

char *input, *input_type, *output, *muoutput, *rmsout, *Niterout, *finalpsi;
long outstpx, outstpy, outstpz;

double omega, edd, h2, h4, q3, q5, norm_psi2, norm_psi3, sx, sy, sz, murel, muend, CC[16], K3;
int QF, QDEPL, GRAD;

long seed = 1973, ADD_RANDOM_PHASE = 0, ADD_VORTICES = 0;
double MAX_PHASE = 1.0, VORTEX_RING_RADIUS = 1.0;

static double complex C1I;
double complex (*e)(double complex);
double complex d_exp(double complex);
double complex c_exp(double complex);

int opt, optimre, optms, MS;
double Na;
long Niter, Nsnap;
long Nx, Ny, Nz;
int rank, nprocs;
long localNx, localNy, offsetNx, offsetNy;
long Nx2, Ny2, Nz2;
double Nad, kN;
double g, gd, g3;
double aho, tau, as, add;
double dx, dy, dz;
double dx2, dy2, dz2;
double dt;
double vnu, vlambda, vgamma;
double par;
double pi, Rcut, Lcut, cutoff;
long cLfact=1, cRfact=1;

double *x, *y, *z;
double *x2, *y2, *z2;
double ***pot;
double ***potdd;
double **cutmem1, *cutmem2, **cutpotdd;
double *moj_cutpotdd;

double complex Ax0, Ay0, Az0, Ax0r, Ay0r, Az0r, Ax, Ay, Az;
double complex *calphax, **calphay, **calphaz;
double complex *cgammax, **cgammay, **cgammaz;
double complex *Aym, *Ayp, *Azm, *Azp;
double complex **ctmpyz;

fftw_plan plan_forward, plan_backward, plan_forward0, plan_backward0;
fftw_plan plan_transpose_x, plan_transpose_y, plan_transpose_dpsi, plan_transpose_Gpsi;



void readpar(void);
void initpsi(double complex ***);
void initpot(void);
void gencoef(void);
void initpotdd(double *, double *, double *, double *, double *, double *);
double cutpotddfun(long, long, double **, double *, double, double);
void calcnorm(double *, double complex ***, double **, double **, double **);
void calcmu(double *, double complex ***, double complex ***, double complex ***, double ***, double complex ***, double ***, fftw_complex *, double **, double **, double **, double **, double **, double **, double complex **, double complex **, double complex ***, double complex ***);
void calcmudet(double *, double complex ***, double complex ***, double complex ***, double ***, double complex ***, double ***, double ***, fftw_complex *, double **, double **, double **, double **, double **, double **, double complex **, double complex **, double ***, double ***, double ***, double *, double complex ***, double complex ***);
void calcpsidd2(double complex ***, double ***, fftw_complex *);
void calcpsidd20(double complex ***, double ***, fftw_complex *);
void calcrms(double *, double complex ***, double complex ***, double **, double **, double **);

void calcnu(double complex ***, double complex ***, double complex ***, double complex ***, double ***, fftw_complex *, double complex **, double complex **);
void calclux(double complex ***, double complex ***, double complex **);
void calcluy(double complex ***, double complex **);
void calcluz(double complex ***, double complex **);

void outdenx(double complex ***, double **, double *, double *, MPI_File);
void outdeny(double complex ***, double **, double *, double *, MPI_File);
void outdenz(double complex ***, double **, double *, double *, MPI_File);
void outdenxy(double complex ***, double ***, double *, MPI_File);
void outdenxz(double complex ***, double ***, double *, MPI_File);
void outdenyz(double complex ***, double ***, double *, MPI_File);
void outpsi2xy(double complex ***, double ***, MPI_File);
void outpsi2x(double complex ***, double **, MPI_File);
void outpsi2y(double complex ***, double **, MPI_File);
void outpsi2xz(double complex ***, double ***, MPI_File);
void outpsi2yz(double complex ***, double ***, MPI_File);
void outdenxyz(double complex ***, double ***, MPI_File);
void outpsi(double complex ***, MPI_File);
void outargxy(double complex ***, double ***, MPI_File);
void outargxz(double complex ***, double ***, MPI_File);
void outargyz(double complex ***, double ***, MPI_File);

//Filon Integral

double f(double, double);
double filon_inner_integral(double x, double c, int base_Ny, double ky)
double double_integral(double b, double c, int base_Nx, int base_Ny, double kx, double ky)
void adaptive_segment_integration(double xi, double xi1, double c, int base_Ny, 
                                 double kx, double ky, double *result, int depth)
double local_oscillation_strength(double x, double y, double kx, double ky)
//void init_thread_local_state();

extern double simpint(double, double *, long);
extern void diff(double, double *, double *, long);
extern void diffc(double, double complex *, double complex *, long);

extern int cfg_init(char *);
extern char *cfg_read(char *);

extern double *alloc_double_vector(long);
extern double complex *alloc_complex_vector(long);
extern double **alloc_double_matrix(long, long);
extern double complex **alloc_complex_matrix(long, long);
extern double ***alloc_double_tensor(long, long, long);
extern double complex ***alloc_complex_tensor(long, long, long);
extern fftw_complex *alloc_fftw_complex_vector(long);
extern void free_double_vector(double *);
extern void free_complex_vector(double complex *);
extern void free_double_matrix(double **);
extern void free_complex_matrix(double complex **);
extern void free_double_tensor(double ***);
extern void free_complex_tensor(double complex ***);
extern void free_fftw_complex_vector(fftw_complex *);
