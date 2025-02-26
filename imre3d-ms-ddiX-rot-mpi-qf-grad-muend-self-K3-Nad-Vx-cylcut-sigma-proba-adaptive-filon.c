/* u
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
 *  * regarding the use of the programs.
 */

#include "imre3d-ms-ddiX-rot-mpi-qf-grad-muend-self-K3-Nad-Vx-cylcut-sigma-proba-adaptive-filon.h"

int main(int argc, char **argv) {
   // Nested parallelism is globally disabled at the start
   //omp_set_nested(0);
   FILE *out;
   FILE *filerms;
   FILE *muout;
   MPI_File mpifile;
   char filename[MAX_FILENAME_SIZE];
   int nthreads, rankNx2;
   long cnti, snap, nsteps, cnte;
   double norm, mu[8], mutot, mutotold, Lx;
   double *rms;
   double complex **cbeta;
   double complex ***psi, ***psi_t, ***dpsi, ***dpsi_t, ***Gpsi, ***Gpsi_t;
   double complex **tmpdpsi1, **tmpdpsi2;
   double ***dpsiL;
   double ***psidd2, ***psidd20;
   double **tmpxi, **tmpyi, **tmpzi, **tmpxj, **tmpyj, **tmpzj, ***tmpxmui, ***tmpymui, ***tmpzmui;
   double **outx, **outy, **outz;
   double ***outxy, ***outxz, ***outyz, **outpsix, **outpsiy;
   double ***outxyz;
   fftw_complex *psidd2fft;

   int provided;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   pi = 3.14159265358979;

   struct timeval start, stop, iter_start, iter_stop;
   double wall_time, init_time, iter_time;
   iter_time = 0.;
   gettimeofday(&start, NULL);

   if ((argc != 3) || (strcmp(*(argv + 1), "-i") != 0)) {
      fprintf(stderr, "Usage: %s -i <input parameter file> \n", *argv);
      exit(EXIT_FAILURE);
   }

   if (! cfg_init(argv[2])) {
      fprintf(stderr, "Wrong input parameter file.\n");
      exit(EXIT_FAILURE);
   }

   readpar();
   srand48(seed);

   if (optimre == 1) {
      C1I = 1. + I * 0.;
      e = &d_exp;
   } else {
      C1I = 0. + I * 1.;
      e = &c_exp;
   }

   //---------------------------------------------------------------------------

   gd *= MS;
   edd = (4. * pi / 3.) * gd / g;
   Nad = Na;
   g3 = K3 * Nad * Nad / pow(aho, 4);

   if (fabs(edd) < 1e-10) {
      q3 = 1.;
      q5 = 1.;
   } else {
      if(fabs(edd - 1.) < 1e-10) {
         q3 = 3. * sqrt(3.) / 4.;
         q5 = 3. * sqrt(3.) / 2.;
      } else {
           //q3 = creal((-(sqrt(3.)*pow(-1. + edd,2)*clog(-3.*(-1. + edd)*edd)) + 2*(sqrt(edd)*(5. + edd)*sqrt(1. + 2.*edd) +
           //     sqrt(3.)*pow(-1. + edd,2)*log(3.*edd + sqrt(3.)*sqrt(edd)*sqrt(1. + 2.*edd))))/ (16.*sqrt(edd)));

           //q5 = creal((6.*sqrt(1. + 2.*edd)*(11. + edd*(4. + 9.*edd)) + (5.*sqrt(3.)*pow(-1. + edd,3)*(clog(-3.*(-1. + edd)*edd) -
           //     2.*log(3.*edd + sqrt(3.)*sqrt(edd)*sqrt(1. + 2.*edd))))/sqrt(edd))/96.);

           q3 = creal((2.*(5. + edd)*csqrt(1. + 2.*edd) + (csqrt(3.)*cpow(-1. + edd,2)*
                      (clog(1. - edd) - 2.*clog(-(csqrt(3.)*csqrt(edd)) + csqrt(1. + 2.*edd))))/csqrt(edd))/16.);

           q5 = creal((6.*csqrt(1. + 2.*edd)*(11. + edd*(4. + 9.*edd)) - (5.*csqrt(3.)*cpow(-1. + edd,3)*(clog(1. - edd) - 
                      2.*clog(-(csqrt(3.)*csqrt(edd)) + csqrt(1. + 2.*edd))))/csqrt(edd))/96.);
      }
   }

//   q5 = 1. + 1.5 * edd * edd;
//   q3 = 1. + 0.3 * edd * edd;
   q3 *= QF * QDEPL;
   q5 *= QF;

   h2 = 32. * sqrt(pi) * pow(as  * BOHR_RADIUS / aho, 2.5) * pow(Nad, 1.5) * (4. * q5 + q3) / 3.;
   h4 = 8. / (3. * sqrt(pi)) * pow(as * BOHR_RADIUS / aho, 1.5) * sqrt(Nad) * q3;


   if ((fabs(edd) >= 1e-10) && (fabs(edd - 1.) >= 1e-10)) {

      CC[1] = creal((-2.*csqrt(edd)*csqrt(1. + 2.*edd)*(-15. + edd*(80. + 49.*edd)) - csqrt(3.)*(-1. + edd)*(5. + edd*(54. + 31.*edd))*clog(1. - edd) +
                    2.*csqrt(3.)*(-1. + edd)*(5. + edd*(54. + 31.*edd))*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd)))/(576.*cpow(edd,1.5)*csqrt(pi)));

      CC[2] = creal((2.*csqrt(edd)*csqrt(1. + 2.*edd)*(-15. + edd*(-250. + 37.*edd)) + csqrt(3.)*cpow(-1. + edd,2.)*(-5. + 43*edd)*clog(1. - edd) -
                    2.*csqrt(3.)*cpow(-1. + edd,2.)*(-5. + 43.*edd)*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd)))/(1152.*cpow(edd,1.5)*csqrt(pi)));

      CC[3] = CC[2];

      CC[4] = creal((-5.*(2.*csqrt(edd)*csqrt(1. + 2.*edd)*(-3. + edd*(-8. + 5.*edd)) + csqrt(3.)*(-1. + edd)*(1. + edd*(6. + 11.*edd))*clog(1. - edd) +
                    2.*csqrt(3.)*(1. + edd*(5. + (5. - 11*edd)*edd))*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd))))/(576.*cpow(edd,1.5)*csqrt(pi)));

      CC[5] = creal((5*(2.*csqrt(edd)*csqrt(1. + 2.*edd)*(-3. + edd*(-2. + 17.*edd)) - csqrt(3.)*cpow(-1. + edd,2.)*(1. + edd)*clog(1. - edd) +
                    2.*csqrt(3.)*cpow(-1. + edd,2.)*(1 + edd)*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd))))/(1152.*cpow(edd,1.5)*csqrt(pi)));

      CC[6] = CC[5];

      CC[7] = creal((5.*(2.*csqrt(edd)*csqrt(1. + 2.*edd)*(-3. + edd*(-8. + 5.*edd)) + csqrt(3.)*(-1. + edd)*(1 + edd*(6. + 11.*edd))*clog(1. - edd) +
                    2.*csqrt(3.)*(1. + edd*(5. + (5. - 11*edd)*edd))*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd))))/(96.*cpow(edd,1.5)*csqrt(pi)));

      CC[8] = creal((5.*(2.*csqrt(edd)*csqrt(1. + 2.*edd)*(3. + (2. - 17.*edd)*edd) + csqrt(3.)*cpow(-1. + edd,2.)*(1. + edd)*clog(1. - edd) -
                    2.*csqrt(3.)*cpow(-1. + edd,2.)*(1. + edd)*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd))))/(192.*cpow(edd,1.5)*csqrt(pi)));

      CC[9] = CC[8];

      CC[10] = creal((2.*csqrt(edd)*csqrt(1. + 2.*edd)*(-15. + edd*(-100. + 13.*edd)) + csqrt(3.)*(-1. + edd)*(5. + edd*(18. + 67.*edd))*clog(1. - edd) -
                     2.*csqrt(3.)*(-1. + edd)*(5. + edd*(18. + 67.*edd))*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd)))/(144.*cpow(edd,1.5)*csqrt(pi)));

      CC[11] = creal((csqrt(3.)*cpow(-1. + edd,2.)*(5. + 29*edd)*clog(1. - edd) - 2.*(csqrt(edd)*csqrt(1. + 2.*edd)*(-15. + edd*(110. + 109.*edd)) +
                     csqrt(3.)*cpow(-1. + edd,2.)*(5. + 29*edd)*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd))))/(288.*cpow(edd,1.5)*csqrt(pi)));

      CC[12] = CC[11];

      CC[13] = creal((5.*(2.*csqrt(edd)*csqrt(1. + 2.*edd)*(-3. + edd*(-8. + 5.*edd)) + csqrt(3.)*(-1. + edd)*(1. + edd*(6. + 11.*edd))*clog(1. - edd) +
                     2.*csqrt(3.)*(1 + edd*(5. + (5. - 11*edd)*edd))*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd))))/(144.*cpow(edd,1.5)*csqrt(pi)));

      CC[14] = creal((-5.*(2.*csqrt(edd)*csqrt(1. + 2.*edd)*(-3. + edd*(-2. + 17.*edd)) - csqrt(3.)*cpow(-1. + edd,2.)*(1. + edd)*clog(1. - edd) +
                     2.*csqrt(3.)*cpow(-1. + edd,2.)*(1. + edd)*clog(csqrt(3.)*csqrt(edd) + csqrt(1. + 2.*edd))))/(288.*cpow(edd,1.5)*csqrt(pi)));

      CC[15] = CC[14];
   }

   if (fabs(edd - 1.) < 1e-10) {

      CC[1] = -19./(16.*sqrt(3.*pi));

      CC[2] = CC[1];

      CC[3] = CC[1];

      CC[4] = 5./(16.*sqrt(3.*pi));

      CC[5] = CC[4];

      CC[6] = CC[4];

      CC[7] = -5.*sqrt(3./pi)/8.;

      CC[8] = CC[7];

      CC[9] = CC[7];

      CC[10] = -17./(4.*sqrt(3.*pi));

      CC[11] = CC[10];

      CC[12] = CC[10];

      CC[13] = -5./(4.*sqrt(3.*pi));

      CC[14] = CC[13];

      CC[15] = CC[13];

   }

   if (fabs(edd) < 1e-10) {

      CC[1] = -53./(72.*sqrt(pi));

      CC[2] = CC[1];

      CC[3] = CC[1];

      CC[4] = -5./(72.*sqrt(pi));

      CC[5] = CC[4];

      CC[6] = CC[4];

      CC[7] = 5./(12.*sqrt(pi));

      CC[8] = CC[7];

      CC[9] = CC[7];

      CC[10] = -19./(18.*sqrt(pi));

      CC[11] = CC[10];

      CC[12] = CC[10];

      CC[13] = 5./(18.*sqrt(pi));

      CC[14] = CC[13];

      CC[15] = CC[13];
   }

   for (cnte = 0; cnte < 16; cnte ++) CC[cnte] *= GRAD * pow(as * BOHR_RADIUS / aho, 1.5) * sqrt(Nad);
   
   //---------------------------------------------------------------------------

   assert(Nx % nprocs == 0);
   assert(Ny % nprocs == 0);

   localNx = Nx / nprocs;
   localNy = Ny / nprocs;
   offsetNx = rank * localNx;
   offsetNy = rank * localNy;

   Nx2 = Nx / 2; Ny2 = Ny / 2; Nz2 = Nz / 2;
   dx2 = dx * dx; dy2 = dy * dy; dz2 = dz * dz;

   rankNx2 = Nx2 / localNx;

   #pragma omp parallel
   #pragma omp master
   nthreads = omp_get_num_threads();

   // Allocation of memory ------------------------------------------

   rms = alloc_double_vector(RMS_ARRAY_SIZE);

   x = alloc_double_vector(Nx);
   y = alloc_double_vector(Ny);
   z = alloc_double_vector(Nz);

   x2 = alloc_double_vector(Nx);
   y2 = alloc_double_vector(Ny);
   z2 = alloc_double_vector(Nz);

//   cLfact = (long) Lcut / Nx;
//   if (cLfact == 0) cLfact = 1;
//   cRfact = (long) ((1. / Rcut) / Ny);
//   if (cRfact == 0) cRfact = 1;

   pot = alloc_double_tensor(localNx, Ny, Nz);
   potdd = alloc_double_tensor(localNy, Nx, Nz);
   cutmem1 = alloc_double_matrix(cLfact * Nx, cRfact * Ny);
   cutmem2 = alloc_double_vector(cLfact * Nx);
   cutpotdd = alloc_double_matrix(Nx, Ny + 1);
   psi = alloc_complex_tensor(localNx, Ny, Nz);
   psi_t = alloc_complex_tensor(Nx, localNy, Nz);
   psidd2 = alloc_double_tensor(localNx, Ny, 2 * (Nz2 + 1));
   if (muoutput != NULL) psidd20 = alloc_double_tensor(localNx, Ny, 2 * (Nz2 + 1));

   dpsi = alloc_complex_tensor(localNx, Ny, Nz);
   dpsi_t = alloc_complex_tensor(Nx, localNy, Nz);
   Gpsi = alloc_complex_tensor(localNx, Ny, Nz);
   Gpsi_t = alloc_complex_tensor(Nx, localNy, Nz);
   dpsiL = alloc_double_tensor(localNx, Ny, Nz);

   calphax = alloc_complex_vector(Nx);
   calphay = alloc_complex_matrix(Nz, Ny);
   calphaz = alloc_complex_matrix(Nz, Ny);
   cbeta = alloc_complex_matrix(nthreads, MAX(Nx, Ny, Nz) - 1);
   cgammax = alloc_complex_vector(Nx);
   cgammay = alloc_complex_matrix(Nz, Ny);
   cgammaz = alloc_complex_matrix(Nz, Ny);
   
   Aym = alloc_complex_vector(Nz);
   Ayp = alloc_complex_vector(Nz);
   Azm = alloc_complex_vector(Ny);
   Azp = alloc_complex_vector(Ny);

   tmpxi = alloc_double_matrix(nthreads, Nx);
   tmpyi = alloc_double_matrix(nthreads, Ny);
   tmpzi = alloc_double_matrix(nthreads, Nz);
   tmpxj = alloc_double_matrix(nthreads, Nx);
   tmpyj = alloc_double_matrix(nthreads, Ny);
   tmpzj = alloc_double_matrix(nthreads, Nz);
   if (muoutput != NULL) tmpxmui = alloc_double_tensor(8, nthreads, Nx);
   if (muoutput != NULL) tmpymui = alloc_double_tensor(8, nthreads, Ny);
   if (muoutput != NULL) tmpzmui = alloc_double_tensor(8, nthreads, Nz);
   tmpdpsi1 = alloc_complex_matrix(nthreads, MAX(Nx, Ny, Nz));
   tmpdpsi2 = alloc_complex_matrix(nthreads, MAX(Nx, Ny, Nz));


   psidd2fft = alloc_fftw_complex_vector(Nx * localNy * (Nz2 + 1));

   outx = alloc_double_matrix(localNx, 2);
   outy = alloc_double_matrix(localNy, 2);
   outz = alloc_double_matrix(Nz, 2); // Because rank 0 will assemble outz
   outxy = alloc_double_tensor(localNx, Ny, 3);
   outpsix=alloc_double_matrix(localNx,2);
   outpsiy=alloc_double_matrix(Ny,2);
   outxz = alloc_double_tensor(localNx, Nz, 3);
   outyz = alloc_double_tensor((rank == rankNx2) ? Ny : localNy, Nz, 3);
   outxyz = dpsiL;
   ctmpyz = alloc_complex_matrix(Ny, Nz);

   // -----------------------------------------------------------------------

   if (opt == 2) par = 2.;
   else par = 1.;

   g = par * g;
   gd = par * gd;
   g3 = par * g3;
   h2 = par * h2;
   tau = par * tau;
   omega = par * omega;
   for (cnte = 0; cnte < 16; cnte ++) CC[cnte] *= par;


   if (input == NULL) {
      if (sx == 0.) {
         if (vgamma > 0.) {
            sx = 1. / sqrt(vgamma);
         } else sx = 1.;
      }

      if (sy == 0.) {  
         if (vnu > 0.) {
            sy = 1. / sqrt(vnu);  
         } else sy = 1.;  
      }

      if (sz == 0.) {
         if (vlambda > 0.) {
            sz = 1. / sqrt(vlambda);
         } else sz = 1.;
      }
   }

   if (rank == 0) {
      if (output != NULL) {
         sprintf(filename, "%s.txt", output);
         out = fopen(filename, "w");
      } else out = stdout;
   } else out = fopen("/dev/null", "w");

   if (rank == 0) {
      if (muoutput != NULL) {
         sprintf(filename, "%s.txt", muoutput);
         muout = fopen(filename, "w");
      } else muout = NULL;
   } else muout = fopen("/dev/null", "w");

   if (rank == 0) {
      if (rmsout != NULL) {
         sprintf(filename, "%s.txt", rmsout);
         filerms = fopen(filename, "w");
      } else filerms = NULL;
   } else filerms = fopen("/dev/null", "w");

   fprintf(out, "\n**********************************************\n");
   if (optimre == 1) {
      fprintf(out, "Imaginary");
   } else {
      fprintf(out, "Real");
   }
   fprintf(out, "-time propagation in 3D, OPTION = %d\nMPI nodes = %d, OMP threads = %d, cores = %d\n", opt, nprocs, nthreads, nprocs * nthreads);
   fprintf(out, "**********************************************\n\nInteractions\n");

   if (cfg_read("G") != NULL) {
      fprintf(out, "Contact: G = %.6le, G * par = %.6le\n", g / par, g);
   } else {
      fprintf(out, "Contact: Natoms = %.11le, as = %.6le * a0, G = %.6le, G * par = %.6le\n", Nad, as, g / par, g);
   }

   fprintf(out, "Three-body: K3 =  %.6le, g3 = %.6le\n", K3, g3);

   if(optms == 0) {
      fprintf(out, "Regular ");
   } else {
      fprintf(out, "Microwave-shielded ");
   }

   if (cfg_read("GDD") != NULL) {
      fprintf(out, "DDI: GD = %.6le, GD * par =  %.6le, edd = %.6le\n", gd / par, gd, edd);
   } else {
        fprintf(out, "DDI: add = %.6le * a0, GD = %.6le, GD * par =  %.6le, edd = %.6le\n", add, gd / par, gd, edd);
   }

   fprintf(out, "     Dipolar cutoffs Rcut = %.6le, Lcut = %.6le, Scut = %.6le,\n\n", Rcut, Lcut,cutoff);

   if (QF == 1) {
      fprintf(out, "QF = 1, QDEPL = %i: h2 = %.6le, h4 = %.6le\n        q3 = %.6le, q5 = %.6le\n\n", QDEPL, h2, h4, q3, q5);
   } else  fprintf(out, "QF = 0\n\n");
   
   fprintf(out, "GRAD = %i\n\n", GRAD);

   fprintf(out, "Trap parameters\nGAMMA = %.6le, NU = %.6le, LAMBDA = %.6le\n", vgamma, vnu, vlambda);
   fprintf(out, "Parameters of rotation: OMEGA = %.6e, OMEGA * par = %.6e\n\n", omega / par, omega);
   fprintf(out, "Space discretization: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);
   fprintf(out, "                      DX = %.6le, DY = %.6le, DZ = %.6le\n", dx, dy, dz);
   if (cfg_read("AHO") != NULL) fprintf(out, "      Unit of length: aho = %.6le m\n", aho);
   fprintf(out, "\nTime discretization: NITER = %li (NSNAP = %li)\n", Niter, Nsnap);
   fprintf(out, "                     DT = %.6le\n\n",  dt);
   fprintf(out, "Initial state: ");
   if (input != NULL) {
      fprintf(out, "file %s\n", input);
   } else {
      fprintf(out, "Gaussian\n               SX = %.6le, SY = %.6le, SZ = %.6le\n", sx, sy, sz);
   }
   fprintf(out, "ADD_VORTICES = %li, ADD_RANDOM_PHASE = %li\n", ADD_VORTICES, ADD_RANDOM_PHASE);                      
   if (ADD_VORTICES > 1) fprintf(out, "VORTEX_RING_RADIUS = %.6le, ", VORTEX_RING_RADIUS);
   if (ADD_RANDOM_PHASE == 1) fprintf(out, "SEED = %li, MAX_PHASE = %.6le", seed, MAX_PHASE);
   fprintf(out, "\n\n");

   if (cfg_read("TAU") != NULL) fprintf(out, "       Unit of time: tau = %.6le s\n", tau);
   fprintf(out, "MUREL = %.6le, MUEND=%.6le\n\n", murel, muend);

   fprintf(out, "-------------------------------------------------------------------\n");
   fprintf(out, "Snap      Na             QD             mu             <r>\n");
   fprintf(out, "-------------------------------------------------------------------\n");
   fflush(out);

   if (muoutput != NULL) {
      fprintf(muout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
      fprintf(muout, "Snap      mu             Kin            Pot            Contact        DDI            Dmu(Contact)   Dmu(DDI)       Dmu(grad)      Rot            Lx  \n");
      fprintf(muout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
      fflush(muout);
   }

   if (rmsout != NULL) {
      fprintf(filerms, "\n**********************************************\n");
      if (optimre == 1) {
         fprintf(filerms, "Imaginary");
      } else {
         fprintf(filerms, "Real");
      }
      fprintf(filerms, "-time propagation in 3D, OPTION = %d\nMPI nodes = %d, OMP threads = %d, cores = %d\n", opt, nprocs, nthreads, nprocs * nthreads);
      fprintf(filerms, "**********************************************\n\nInteractions\n");

      if (cfg_read("G") != NULL) {
         fprintf(filerms, "Contact: G = %.6le, G * par = %.6le\n", g / par, g);
      } else {
         fprintf(filerms, "Contact: Natoms = %.11le, as = %.6le * a0, G = %.6le, G * par = %.6le\n", Nad, as, g / par, g);
      }

      fprintf(filerms, "Three-body: K3 =  %.6le, g3 = %.6le\n", K3, g3);

      if(optms == 0) {
         fprintf(filerms, "Regular ");
      } else {
         fprintf(filerms, "Microwave-shielded ");
      }

      if (cfg_read("GDD") != NULL) {
         fprintf(filerms, "DDI: GD = %.6le, GD * par = %.6le, edd = %.6le\n", gd / par, gd, edd);
      } else {
           fprintf(filerms, "DDI: add = %.6le * a0, GD = %.6le, GD * par = %.6le, edd = %.6le\n", add, gd / par, gd, edd);
      }

      fprintf(filerms, "     Dipolar cutoff Rcut = %.6le, Lcut = %.6le,Scut = %.6le,\n\n", Rcut, Lcut,cutoff);

      if (QF == 1) {
         fprintf(filerms, "QF = 1: h2 = %.6le, h4 = %.6le\n        q3 = %.6le, q5 = %.6le\n\n", h2, h4, q3, q5);
      } else  fprintf(filerms, "QF = 0\n\n");

      fprintf(filerms, "GRAD = %i\n\n", GRAD);

      fprintf(filerms, "Trap parameters:\nGAMMA = %.6le, NU = %.6le, LAMBDA = %.6le\n", vgamma, vnu, vlambda);
      fprintf(filerms, "Parameters of rotation: OMEGA = %.6e, OMEGA * par = %.6e\n\n", omega / par, omega);
      fprintf(filerms, "Space discretization: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);
      fprintf(filerms, "                      DX = %.6le, DY = %.6le, DZ = %.6le\n", dx, dy, dz);
      if (cfg_read("AHO") != NULL) fprintf(filerms, "      Unit of length: aho = %.6le m\n", aho);
      fprintf(filerms, "\nTime discretization: NITER = %li (NSNAP = %li)\n", Niter, Nsnap);
      fprintf(filerms, "                     DT = %.6le\n\n",  dt);
      fprintf(filerms, "Initial state: ");
      if (input != NULL) {
         fprintf(filerms, "file %s\n", input);
      } else {
         fprintf(filerms, "Gaussian\n               SX = %.6le, SY = %.6le, SZ = %.6le\n", sx, sy, sz);
      }
      fprintf(filerms, "ADD_VORTICES = %li, ADD_RANDOM_PHASE = %li\n", ADD_VORTICES, ADD_RANDOM_PHASE);
      if (ADD_VORTICES > 1) fprintf(filerms, "VORTEX_RING_RADIUS = %.6le, ", VORTEX_RING_RADIUS);
      if (ADD_RANDOM_PHASE == 1) fprintf(filerms, "SEED = %li, MAX_PHASE = %.6le", seed, MAX_PHASE);
      fprintf(filerms, "\n\n");

      if (cfg_read("TAU") != NULL) fprintf(filerms, "       Unit of time: tau = %.6le s\n", tau);
      fprintf(filerms, "MUREL = %.6le, MUEND=%.6le\n\n", murel, muend);


      fprintf(filerms, "-------------------------------------------------------------------\n");
      fprintf(filerms, "Snap      <r>            <x>            <y>            <z>\n");
      fprintf(filerms, "-------------------------------------------------------------------\n");
      fflush(filerms);
   }

   fftw_init_threads();
   fftw_mpi_init();
   fftw_plan_with_nthreads(nthreads);

   long n[] = { Nx, Ny, Nz };
   plan_forward = fftw_mpi_plan_many_dft_r2c(3, n, 1, localNx, localNy, psidd2[0][0], psidd2fft, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_OUT);
   plan_backward = fftw_mpi_plan_many_dft_c2r(3, n, 1, localNy, localNx, psidd2fft, psidd2[0][0], MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_IN);

   if (muoutput != NULL) plan_forward0 = fftw_mpi_plan_many_dft_r2c(3, n, 1, localNx, localNy, psidd20[0][0], psidd2fft, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_OUT);
   if (muoutput != NULL) plan_backward0 = fftw_mpi_plan_many_dft_c2r(3, n, 1, localNy, localNx, psidd2fft, psidd20[0][0], MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_IN);

   plan_transpose_x = fftw_mpi_plan_many_transpose(Nx, Ny * Nz, 2, localNx, localNy * Nz, (double *) **psi, (double *) **psi_t, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_OUT);
   plan_transpose_y = fftw_mpi_plan_many_transpose(Ny * Nz, Nx, 2, localNy * Nz, localNx, (double *) **psi_t, (double *) **psi, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_IN);
   plan_transpose_dpsi = fftw_mpi_plan_many_transpose(Ny * Nz, Nx, 2, localNy * Nz, localNx, (double *) **dpsi_t, (double *) **dpsi, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_IN);
   plan_transpose_Gpsi = fftw_mpi_plan_many_transpose(Ny * Nz, Nx, 2, localNy * Nz, localNx, (double *) **Gpsi_t, (double *) **Gpsi, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_IN);


   initpsi(psi);
   initpot();
   gencoef();
   // Enable nested parallelism for this section
   //omp_set_nested(1);
   initpotdd(*tmpxi, *tmpyi, *tmpzi, *tmpxj, *tmpyj, *tmpzj);
   //omp_set_nested(0); // Disable nested parallelism after this section

   calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
   calcrms(rms, psi, psi_t, tmpxi, tmpyi, tmpzi);

   if (muoutput != NULL) {
      calcmudet(mu, psi, psi_t, dpsi, dpsiL, dpsi_t, psidd2, psidd20, psidd2fft, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpdpsi1, tmpdpsi2, tmpxmui, tmpymui, tmpzmui, &Lx, Gpsi, Gpsi_t);
      // if(rank==0){
      // printf("Mu = %.26le, par = %.5le, norm_psi2 = %.26le",mu[2], par, norm_psi2);}
      mutot = 0.;
      for (cnte = 0; cnte < 8; cnte ++) {
         mu[cnte] /= par * norm_psi2;
         mutot += mu[cnte];
      }
      //if(rank==0){
      // printf("Mu = %.26le, par = %.5le, norm_psi2 = %.26le",mu[2], par, norm_psi2);}
      Lx /= norm_psi2;

      fprintf(muout, "%-9d %-14.6le %-14.26le %-14.6le %-14.26le %-14.26le %-14.6le %-14.6le %-14.6le %-14.6le %-14.6le\n", 0, mutot, mu[0], mu[1], mu[2], mu[3], mu[4], mu[5], mu[6], mu[7], Lx);
      fflush(muout);
   } else {
      calcmu(&mutot, psi, psi_t, dpsi, dpsiL, dpsi_t, psidd2, psidd2fft, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpdpsi1, tmpdpsi2, Gpsi, Gpsi_t);
      mutot /= par * norm_psi2;
   }

   MPI_Bcast(&mutot, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   fprintf(out, "%-9d %-19.11le %-19.11le %-19.11le %-19.11le\n", 0, Nad, h4 * norm_psi3, mutot, *rms / sqrt(norm_psi2));
   fflush(out);

   mutotold = mutot;

   if (rmsout != NULL) {
      fprintf(filerms, "%-9d %-19.11le %-19.26le %-19.26le %-19.26le\n", 0, rms[0] / sqrt(norm_psi2), rms[1] / sqrt(norm_psi2), rms[2] / sqrt(norm_psi2), rms[3] / sqrt(norm_psi2));
      fflush(filerms);
   }

   if (Niterout != NULL) {

      char itername[10];
      sprintf(itername, "-%06d-", 0);

      if (outflags & DEN_X00) {
            sprintf(filename, "%s%s1d_x00.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outpsi2x(psi, outpsix, mpifile);
            MPI_File_close(&mpifile);
            sprintf(filename, "%s%s1d_0y0.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outpsi2y(psi, outpsiy, mpifile);
            MPI_File_close(&mpifile);
         }
      
      if (outflags & DEN_XYZ) {
         sprintf(filename, "%s%s3d.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenxyz(psi, outxyz, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & DEN_X) {
         sprintf(filename, "%s%s1d_x.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenx(psi, outx, *tmpyi, *tmpzi, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & DEN_Y) {
         sprintf(filename, "%s%s1d_y.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdeny(psi_t, outy, *tmpxi, *tmpzi, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & DEN_Z) {
         sprintf(filename, "%s%s1d_z.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenz(psi, outz, *tmpxi, *tmpyi, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & DEN_XY) {
         sprintf(filename, "%s%s2d_xy.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenxy(psi, outxy, *tmpzi, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & DEN_XZ) {
         sprintf(filename, "%s%s2d_xz.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenxz(psi, outxz, *tmpyi, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & DEN_YZ) {
          sprintf(filename, "%s%s2d_yz.bin", Niterout, itername);
          MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
          outdenyz(psi_t, outyz, *tmpxi, mpifile);
          MPI_File_close(&mpifile);
      }

      if (outflags & DEN_XY0) {
         sprintf(filename, "%s%s3d_xy0.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outpsi2xy(psi, outxy, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & DEN_X0Z) {
         sprintf(filename, "%s%s3d_x0z.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outpsi2xz(psi, outxz, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & DEN_0YZ) {
         sprintf(filename, "%s%s3d_0yz.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outpsi2yz(psi, outyz, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & ARG_XY0) {
         sprintf(filename, "%s-phase%s3d_xy0.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outargxy(psi, outxy, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & ARG_X0Z) {
         sprintf(filename, "%s-phase%s3d_x0z.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outargxz(psi, outxz, mpifile);
         MPI_File_close(&mpifile);
      }

      if (outflags & ARG_0YZ) {
         sprintf(filename, "%s-phase%s3d_0yz.bin", Niterout, itername);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outargyz(psi, outyz, mpifile);
         MPI_File_close(&mpifile);
      }
   }

   //  -----------------------------------------------------------NITER

   gettimeofday(&iter_start, NULL);

   nsteps = Niter / Nsnap;
   for (snap = 1; snap < Nsnap; snap ++) {
      for (cnti = 0; cnti < nsteps; cnti ++) {
         calcnu(psi, psi_t, Gpsi, Gpsi_t, psidd2, psidd2fft, tmpdpsi1, tmpdpsi2);
         calclux(psi, psi_t, cbeta);
         calcluy(psi, cbeta);
         calcluz(psi, cbeta);
         calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
      }

      calcrms(rms, psi, psi_t, tmpxi, tmpyi, tmpzi);

      if (muoutput != NULL) {
         calcmudet(mu, psi, psi_t, dpsi, dpsiL, dpsi_t, psidd2, psidd20, psidd2fft, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpdpsi1, tmpdpsi2, tmpxmui, tmpymui, tmpzmui, &Lx, Gpsi, Gpsi_t);

         mutot = 0.;
         for (cnte = 0; cnte < 8; cnte ++) {
            mu[cnte] /= par * norm_psi2;
            mutot += mu[cnte];
         }

         Lx /= norm_psi2;

         fprintf(muout, "%-9li %-14.6le %-14.26le %-14.6le %-14.26le %-14.26le %-14.6le %-14.6le %-14.6le %-14.6le %-14.6le\n", snap, mutot, mu[0], mu[1], mu[2], mu[3], mu[4], mu[5], mu[6], mu[7], Lx);
         fflush(muout);
      } else {
         calcmu(&mutot, psi, psi_t, dpsi, dpsiL, dpsi_t, psidd2, psidd2fft, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj, tmpdpsi1, tmpdpsi2, Gpsi, Gpsi_t);
         mutot /= par * norm_psi2;
      }

      MPI_Bcast(&mutot, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      fprintf(out, "%-9li %-19.11le %-19.11le %-19.11le %-19.11le\n", snap, Nad, h4 * norm_psi3, mutot, *rms / sqrt(norm_psi2));
      fflush(out);

      if (rmsout != NULL) {
         fprintf(filerms, "%-9li %-19.11le %-19.26le %-19.26le %-19.26le\n", snap, rms[0] / sqrt(norm_psi2), rms[1] / sqrt(norm_psi2), rms[2] / sqrt(norm_psi2), rms[3] / sqrt(norm_psi2));
         fflush(filerms);
      }

      if (Niterout != NULL) {

         char itername[10];
         sprintf(itername, "-%06li-", snap);

         if (outflags & DEN_X00) {
            sprintf(filename, "%s%s1d_x00.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outpsi2x(psi, outx, mpifile);
            MPI_File_close(&mpifile);
            sprintf(filename, "%s%s1d_0y0.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outpsi2y(psi, outpsiy, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & DEN_XYZ) {
            sprintf(filename, "%s%s3d.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outdenxyz(psi, outxyz, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & DEN_X) {
            sprintf(filename, "%s%s1d_x.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outdenx(psi, outx, *tmpyi, *tmpzi, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & DEN_Y) {
            sprintf(filename, "%s%s1d_y.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outdeny(psi_t, outy, *tmpxi, *tmpzi, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & DEN_Z) {
            sprintf(filename, "%s%s1d_z.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outdenz(psi, outz, *tmpxi, *tmpyi, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & DEN_XY) {
            sprintf(filename, "%s%s2d_xy.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outdenxy(psi, outxy, *tmpzi, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & DEN_XZ) {
            sprintf(filename, "%s%s2d_xz.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outdenxz(psi, outxz, *tmpyi, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & DEN_YZ) {
             sprintf(filename, "%s%s2d_yz.bin", Niterout, itername);
             MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
             outdenyz(psi_t, outyz, *tmpxi, mpifile);
             MPI_File_close(&mpifile);
         }

         if (outflags & DEN_XY0) {
            sprintf(filename, "%s%s3d_xy0.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outpsi2xy(psi, outxy, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & DEN_X0Z) {
            sprintf(filename, "%s%s3d_x0z.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outpsi2xz(psi, outxz, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & DEN_0YZ) {
            sprintf(filename, "%s%s3d_0yz.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outpsi2yz(psi, outyz, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & ARG_XY0) {
            sprintf(filename, "%s-phase%s3d_xy0.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outargxy(psi, outxy, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & ARG_X0Z) {
            sprintf(filename, "%s-phase%s3d_x0z.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outargxz(psi, outxz, mpifile);
            MPI_File_close(&mpifile);
         }

         if (outflags & ARG_0YZ) {
            sprintf(filename, "%s-phase%s3d_0yz.bin", Niterout, itername);
            MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
            outargyz(psi, outyz, mpifile);
            MPI_File_close(&mpifile);
         }
      }

      if (fabs((mutotold - mutot) / mutot) < murel) break;
      mutotold = mutot;

      if (mutot > muend) break;
   }

   gettimeofday(&iter_stop, NULL);
   iter_time += (double) (((iter_stop.tv_sec - iter_start.tv_sec) * 1000 + (iter_stop.tv_usec - iter_start.tv_usec)/1000.0) + 0.5);

   fprintf(out, "-------------------------------------------------------------------\n");
   fflush(out);
   fprintf(muout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n");
   fflush(muout);

   if (rmsout != NULL) {
      fprintf(filerms, "-------------------------------------------------------------------\n\n");
      fclose(filerms);
   }

   if (finalpsi != NULL) {
       sprintf(filename, "%s.bin", finalpsi);
       MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
       outpsi(psi, mpifile);
       MPI_File_close(&mpifile);
   }

   // Free all dynamically allocated memory. ---------------

   free_double_vector(rms);

   free_double_vector(x);
   free_double_vector(y);
   free_double_vector(z);

   free_double_vector(x2);
   free_double_vector(y2);
   free_double_vector(z2);

   free_double_tensor(pot);
   free_double_tensor(potdd);
   free_double_matrix(cutmem1);
   free_double_vector(cutmem2);
   free_double_matrix(cutpotdd);
   free_complex_tensor(psi);
   free_complex_tensor(psi_t);
   free_double_tensor(psidd2);
   if (muoutput != NULL) free_double_tensor(psidd20);

   free_complex_tensor(dpsi);
   free_complex_tensor(dpsi_t);
   free_double_tensor(dpsiL);

   free_complex_vector(calphax);
   free_complex_matrix(calphay);
   free_complex_matrix(calphaz);
   free_complex_matrix(cbeta);
   free_complex_vector(cgammax);
   free_complex_matrix(cgammay);
   free_complex_matrix(cgammaz);
   
   free_complex_vector(Aym);
   free_complex_vector(Ayp);
   free_complex_vector(Azm);
   free_complex_vector(Azp);

   free_double_matrix(tmpxi);
   free_double_matrix(tmpyi);
   free_double_matrix(tmpzi);
   free_double_matrix(tmpxj);
   free_double_matrix(tmpyj);
   free_double_matrix(tmpzj);
   if (muoutput != NULL) free_double_tensor(tmpxmui);
   if (muoutput != NULL) free_double_tensor(tmpymui);
   if (muoutput != NULL) free_double_tensor(tmpzmui);
   free_complex_matrix(tmpdpsi1);
   free_complex_matrix(tmpdpsi2);

   fftw_destroy_plan(plan_forward);
   fftw_destroy_plan(plan_backward);
   if (muoutput != NULL) fftw_destroy_plan(plan_forward0);
   if (muoutput != NULL) fftw_destroy_plan(plan_backward0);
   fftw_destroy_plan(plan_transpose_x);
   fftw_destroy_plan(plan_transpose_y);
   fftw_destroy_plan(plan_transpose_dpsi);

   free_fftw_complex_vector(psidd2fft);

   free_double_matrix(outx);
   free_double_matrix(outy);
   free_double_matrix(outz);
   free_double_matrix(outpsix);
   free_double_matrix(outpsiy);
   free_double_tensor(outxy);
   free_double_tensor(outxz);
   free_double_tensor(outyz);
   free_complex_matrix(ctmpyz);

   fftw_mpi_cleanup();

   // ----------------------------------------------------

   MPI_Finalize();

   gettimeofday(&stop, NULL);
   wall_time = (double) (((stop.tv_sec - start.tv_sec) * 1000 + (stop.tv_usec - start.tv_usec)/1000.0) + 0.5);
   init_time = wall_time - iter_time;
   fprintf(out, "\nInitialization/allocation wall-clock time: %.3f seconds\n", init_time / 1000.);
   fprintf(out, "Calculation (iterations) wall-clock time : %.3f seconds\n\n", iter_time / 1000.);

   if(output != NULL) fclose(out);
   if(muoutput != NULL) fclose(muout);

   return (EXIT_SUCCESS);
}

double complex d_exp(double complex x) {
    return cexp(x);
}

double complex c_exp(double complex x) {
    return cexp(I * x);
}

/**
 *    Reading input parameters from the configuration file.
 */
void readpar(void) {
   char *cfg_tmp;

   if ((cfg_tmp = cfg_read("OPTION")) == NULL) {
      fprintf(stderr, "OPTION is not defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }
   opt = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("OPTION_IMAGINARY_REAL")) == NULL) {
      fprintf(stderr, "OPTION_IMAGINARY_REAL is not defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }
   optimre = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("OPTION_MICROWAVE_SHIELDING")) == NULL) {
      fprintf(stderr, "OPTION_MICROWAVE_SHIELDING is not defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }
   optms = atol(cfg_tmp);

   if (optms == 0) {
      MS = 1;
   } else {
      MS = - 1;
   }

   if ((cfg_tmp = cfg_read("GRAD")) == NULL) {
      fprintf(stderr, "GRAD is not defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }
   GRAD = atol(cfg_tmp);


   if ((cfg_tmp = cfg_read("NATOMS")) == NULL) {
      fprintf(stderr, "NATOMS is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Na = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("AHO")) == NULL) {
      fprintf(stderr, "AHO is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   aho = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("G")) == NULL) {
      if ((cfg_tmp = cfg_read("AS")) == NULL) {
         fprintf(stderr, "AS is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      as = atof(cfg_tmp);

      g = 4. * pi * as * Na * BOHR_RADIUS / aho;
   } else {
      g = atof(cfg_tmp);
   }

   if ((cfg_tmp = cfg_read("GDD")) == NULL) {
      if ((cfg_tmp = cfg_read("ADD")) == NULL) {
         fprintf(stderr, "ADD is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      add = atof(cfg_tmp);

      gd = 3. * add * Na * BOHR_RADIUS / aho;
   } else {
      gd = atof(cfg_tmp);
   }

   if ((cfg_tmp = cfg_read("K3")) == NULL) {
      K3 = 0.;
   } else K3 = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("OMEGA")) == NULL) {
      omega = 0.;
   } else omega = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("TAU")) != NULL) tau = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("SEED")) != NULL) seed = atol(cfg_tmp);
   if ((cfg_tmp = cfg_read("ADD_RANDOM_PHASE")) != NULL) ADD_RANDOM_PHASE = atol(cfg_tmp);
   if ((cfg_tmp = cfg_read("MAX_PHASE")) != NULL) MAX_PHASE = atof(cfg_tmp);
   if ((cfg_tmp = cfg_read("ADD_VORTICES")) != NULL) ADD_VORTICES = atol(cfg_tmp);
   if ((cfg_tmp = cfg_read("VORTEX_RING_RADIUS")) != NULL) VORTEX_RING_RADIUS = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("QF")) == NULL) {
      QF = 0;
   } else QF = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("QDEPL")) == NULL) {
      QDEPL = 0;
   } else QDEPL = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("SX")) == NULL) sx = 0.;
   else sx = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("SY")) == NULL) sy = 0.;
   else sy = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("SZ")) == NULL) sz = 0.;
   else sz = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("NX")) == NULL) {
      fprintf(stderr, "NX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nx = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NY")) == NULL) {
      fprintf(stderr, "NY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Ny = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NZ")) == NULL) {
      fprintf(stderr, "Nz is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nz = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("DX")) == NULL) dx = sx * 10. * 2 / Nx;
   else dx = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("DY")) == NULL) dy = sy * 10. * 2 / Ny;
   else dy = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("DZ")) == NULL) dz = sz * 10. * 2 / Nz;
   else dz = atof(cfg_tmp);

   // if ((cfg_tmp = cfg_read("BX")) != NULL) dx = atof(cfg_tmp) * 2 / Nx;

   // if ((cfg_tmp = cfg_read("BY")) != NULL) dy = atof(cfg_tmp) * 2 / Ny;

   // if ((cfg_tmp = cfg_read("BZ")) != NULL) dz = atof(cfg_tmp) * 2 / Nz;


  // if ((cfg_tmp = cfg_read("DX")) == NULL) {
     // fprintf(stderr, "DX is not defined in the configuration file.\n");
    //  exit(EXIT_FAILURE);
  // }
   //dx = atof(cfg_tmp);

  // if ((cfg_tmp = cfg_read("DY")) == NULL) {
    //  fprintf(stderr, "DY is not defined in the configuration file.\n");
     // exit(EXIT_FAILURE);
  // }
  // dy = atof(cfg_tmp);

   //if ((cfg_tmp = cfg_read("DZ")) == NULL) {
     // fprintf(stderr, "DZ is not defined in the configuration file.\n");
      //exit(EXIT_FAILURE);
  // }
  // dz = atof(cfg_tmp);


   if ((cfg_tmp = cfg_read("DT")) == NULL) {
      if (optms == 0) dt = dy * dy / 2; else dt = dx * dx / 2;
   } else dt = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("MUREL")) == NULL) {
      fprintf(stderr, "MUREL is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   murel = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("MUEND")) == NULL) {
      fprintf(stderr, "MUEND is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   muend = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("GAMMA")) == NULL) {
      fprintf(stderr, "GAMMA is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vgamma = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("NU")) == NULL) {
      fprintf(stderr, "NU is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vnu = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("LAMBDA")) == NULL) {
      fprintf(stderr, "LAMBDA is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vlambda = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("NITER")) == NULL) {
      fprintf(stderr, "NITER is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Niter = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("CUTOFF")) == NULL) cutoff=Nx * dx / 2;
   else cutoff = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("NSNAP")) == NULL) Nsnap = 1;
   else Nsnap = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("RCUT")) == NULL) Rcut = Ny * dy / 2.;
   else Rcut = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("LCUT")) == NULL) Lcut = Nx * dx / 2;
   else Lcut = atof(cfg_tmp);  

   input = cfg_read("INPUT");
   input_type = cfg_read("INPUT_TYPE");
   output = cfg_read("OUTPUT");
   muoutput = cfg_read("MUOUTPUT");
   rmsout = cfg_read("RMSOUT");
   Niterout = cfg_read("NITEROUT");
   finalpsi = cfg_read("FINALPSI");

   if (Niterout != NULL) {
      if ((cfg_tmp = cfg_read("OUTFLAGS")) == NULL) {
         fprintf(stderr, "OUTFLAGS is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outflags = atoi(cfg_tmp);           
   } else outflags = 0;
 
   if ((Niterout != NULL) || (finalpsi != NULL)) {
      if ((cfg_tmp = cfg_read("OUTSTPX")) == NULL) {
         fprintf(stderr, "OUTSTPX is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpx = atol(cfg_tmp);

      if ((cfg_tmp = cfg_read("OUTSTPY")) == NULL) {
         fprintf(stderr, "OUTSTPY is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpy = atol(cfg_tmp);

      if ((cfg_tmp = cfg_read("OUTSTPZ")) == NULL) {
         fprintf(stderr, "OUTSTPZ is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpz = atol(cfg_tmp);
   }

   return;
}

/**
 *    Initialization of the space mesh and the initial wave function.
 *    psi - array with the wave function values
 *    tmpz - temporary array
 */
void initpsi(double complex ***psi) {
   long cnti, cntj, cntk, cntv;
   double cpsi;
   double tmp, *tmpr, *tmpc, yv, zv;
   MPI_Offset fileoffset;
   MPI_File file = MPI_FILE_NULL;

   tmpr = alloc_double_vector(Nz);
   tmpc = alloc_double_vector(2 * Nz);

   #pragma omp parallel for private(cnti)
   for (cnti = 0; cnti < Nx; cnti ++) {
      x[cnti] = (cnti - Nx2) * dx;
      x2[cnti] = x[cnti] * x[cnti];
   }

   #pragma omp parallel for private(cntj)
   for (cntj = 0; cntj < Ny; cntj ++) {
      y[cntj] = (cntj - Ny2) * dy;
      y2[cntj] = y[cntj] * y[cntj];
   }

   #pragma omp parallel for private(cntk)
   for (cntk = 0; cntk < Nz; cntk ++) {
      z[cntk] = (cntk - Nz2) * dz;
      z2[cntk] = z[cntk] * z[cntk];
   }

   if (input != NULL) {
      MPI_File_open(MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

      if (file == MPI_FILE_NULL) {
         fprintf(stderr, "Specify the proper input file with the initial wave function.\n");
         MPI_Finalize();
         exit(EXIT_FAILURE);
      }

      if (strcmp(input_type, "DEN") == 0) {
         fileoffset = rank * sizeof(double) * localNx * Ny * Nz;
         for (cnti = 0; cnti < localNx; cnti ++) {
            for (cntj = 0; cntj < Ny; cntj ++) {
               MPI_File_read_at(file, fileoffset, tmpr, Nz, MPI_DOUBLE, MPI_STATUS_IGNORE);
               for (cntk = 0; cntk < Nz; cntk ++) {
                  psi[cnti][cntj][cntk] = sqrt(tmpr[cntk]);
               }
               fileoffset += Nz * sizeof(double);
            }
         }
      } else {
         if (strcmp(input_type, "PSI") == 0) {
            fileoffset = rank * sizeof(double complex) * localNx * Ny * Nz;
            for (cnti = 0; cnti < localNx; cnti ++) {
               for (cntj = 0; cntj < Ny; cntj ++) {
                  MPI_File_read_at(file, fileoffset, tmpc, 2 * Nz, MPI_DOUBLE, MPI_STATUS_IGNORE);
                  for (cntk = 0; cntk < Nz; cntk ++) {
                     psi[cnti][cntj][cntk] = tmpc[2 * cntk] + I * tmpc[2 * cntk + 1];
                  }
                  fileoffset += Nz * sizeof(double complex);
               }
            }
         } else {
            fprintf(stderr, "Specify the proper input_type for the file with the initial wave function.\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
         }
      }

      MPI_File_close(&file);
   } else {

      cpsi = sqrt(2. * pi * sqrt(2. * pi) * sx * sy * sz);

      #pragma omp parallel for private(cnti, cntj, cntk, tmp)
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmp = exp(- 0.25 * (x2[offsetNx + cnti] / (sx * sx) + y2[cntj] / (sy * sy) + z2[cntk] / (sz * sz)));
               psi[cnti][cntj][cntk] = tmp / cpsi;
            }
         }
      }
   }

   if ((ADD_RANDOM_PHASE == 1) || (ADD_VORTICES >= 1)) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            ctmpyz[cntj][cntk] = 1.;
            if(ADD_RANDOM_PHASE == 1) ctmpyz[cntj][cntk] *= cexp(2. * pi * MAX_PHASE * I * drand48());
            if(ADD_VORTICES >= 1) ctmpyz[cntj][cntk] *= (y[cntj] + I * z[cntk]);
         }
      }

     if (ADD_VORTICES > 1) {
        for (cntv = 0; cntv < ADD_VORTICES - 1; cntv ++) {
           yv = VORTEX_RING_RADIUS * cos(2. * pi * cntv / (ADD_VORTICES - 1));
           zv = VORTEX_RING_RADIUS * sin(2. * pi * cntv / (ADD_VORTICES - 1));
           for (cntj = 0; cntj < Ny; cntj ++) {
              for (cntk = 0; cntk < Nz; cntk ++) ctmpyz[cntj][cntk] *= (y[cntj] - yv + I * (z[cntk] - zv));
           }
       }
     }

      #pragma omp parallel for private(cnti, cntj, cntk, tmp)
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               psi[cnti][cntj][cntk] *= ctmpyz[cntj][cntk];
            }
         }
      }
   }

   free_double_vector(tmpr);
   free_double_vector(tmpc);

   return;
}

/**
 *    Initialization of the potential.
 */
void initpot() {
   long cnti, cntj, cntk;
   double vgamma2, vnu2, vlambda2;

   vgamma2 = vgamma * vgamma;
   vnu2 = vnu * vnu;
   vlambda2 = vlambda * vlambda;

   #pragma omp parallel for private(cnti, cntj, cntk)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            pot[cnti][cntj][cntk] = 0.5 * par * (vgamma2 * x2[offsetNx + cnti] + vnu2 * y2[cntj] + vlambda2 * z2[cntk]);
         }
      }
   }

   return;
}

/**
 *    Crank-Nicolson scheme coefficients generation.
 */
void gencoef(void) {
   long cnti, cntj, cntk;
   double complex A0, Atmpy, Atmpz, cdt;

   cdt = C1I * dt;

   Ax0 = 1. + cdt / dx2 / (3. - par);
   Ay0 = 1. + cdt / dy2 / (3. - par);
   Az0 = 1. + cdt / dz2 / (3. - par);

   Ax0r = 1. - cdt / dx2 / (3. - par);
   Ay0r = 1. - cdt / dy2 / (3. - par);
   Az0r = 1. - cdt / dz2 / (3. - par);

   Ax = - 0.5 * cdt / dx2 / (3. - par);
   Ay = - 0.5 * cdt / dy2 / (3. - par);
   Az = - 0.5 * cdt / dz2 / (3. - par);

   A0 = 0.5 * I * cdt * omega;

   Atmpy = A0 / (2. * dy);
   Atmpz = A0 / (2. * dz);

   for (cntk = 0; cntk < Nz; cntk ++) {
       Aym[cntk] = Ay + Atmpy * z[cntk];
       Ayp[cntk] = Ay - Atmpy * z[cntk];
   }
   
   for (cntj = 0; cntj < Ny; cntj ++) {
       Azm[cntj] = Az - Atmpz * y[cntj];
       Azp[cntj] = Az + Atmpz * y[cntj];
   }
   
   calphax[Nx - 2] = 0.;
   cgammax[Nx - 2] = - 1. / Ax0;
   for (cnti = Nx - 2; cnti > 0; cnti --) {
      calphax[cnti - 1] = Ax * cgammax[cnti];
      cgammax[cnti - 1] = - 1. / (Ax0 + Ax * calphax[cnti - 1]);
   }

   for (cntk = 0; cntk < Nz; cntk ++) {
      calphay[cntk][Ny - 2] = 0.;
      cgammay[cntk][Ny - 2] = - 1. / Ay0;
      for (cntj = Ny - 2; cntj > 0; cntj --) {
         calphay[cntk][cntj - 1] = Aym[cntk] * cgammay[cntk][cntj];
         cgammay[cntk][cntj - 1] = - 1. / (Ay0 + Ayp[cntk] * calphay[cntk][cntj - 1]);
      }
   }

   for (cntj = 0; cntj < Ny; cntj ++) {
      calphaz[Nz - 2][cntj] = 0.;
      cgammaz[Nz - 2][cntj] = - 1. / Az0;
      for (cntk = Nz - 2; cntk > 0; cntk --) {
         calphaz[cntk - 1][cntj] = Azm[cntj] * cgammaz[cntk][cntj];
         cgammaz[cntk - 1][cntj] = - 1. / (Az0 + Azp[cntj] * calphaz[cntk - 1][cntj]);
      }
   }

   return;
}

// Define the function f(x, y)
inline double f(double x, double y) {
    double x2y2 = x*x*y*y;
    return (1. - 2. * x2y2) / pow(1. + x2y2, 2.5);
}

// Determine local oscillation strength to guide adaptive discretization
inline double local_oscillation_strength(double x, double y, double kx, double ky) {
    // Estimate local frequency based on wave number and coordinate values
    double freq_x = kx / (x * x); // Higher frequency at smaller x values due to Bessel
    double freq_y = ky;           // Constant in y direction
    
    // Also consider function variation
    double delta = 1e-4;
    double df_dx = (f(x+delta, y) - f(x, y)) / delta;
    double df_dy = (f(x, y+delta) - f(x, y)) / delta;
    
    // Combine measures
    return sqrt(freq_x*freq_x + freq_y*freq_y) + sqrt(df_dx*df_dx + df_dy*df_dy);
}

// Simple structure to track recursion depth per thread
typedef struct {
    int depth;
} ThreadState;

// Define thread-local variable
static ThreadState thread_state;
#pragma omp threadprivate(thread_state)

// Filon-Type Quadrature for the inner integral (over y) with adaptive sampling
double filon_inner_integral(double x, double c, int base_Ny, double ky) {
    // Determine adaptive sampling based on x
    int adaptive_Ny = base_Ny;
    double oscil_strength = fabs(ky) + 10.0 / (1.0 + x*x); 
    
    // Increase sampling for higher oscillations
    if (oscil_strength > 5.0) {
        adaptive_Ny = (int)(base_Ny * (1.0 + log(oscil_strength/5.0) * 2.0));
    }
    
    double h = c / (adaptive_Ny - 1); // Step size
    double sum_cos = 0.0;
   int i;
    #pragma omp parallel for reduction(+:sum_cos) if(adaptive_Ny > 128)
    for (i = 0; i < adaptive_Ny - 1; i++) {
        double yi = i * h;
        double yi1 = (i + 1) * h;
        double ym = (yi + yi1) / 2.0; // Midpoint

        // Calculate function values directly
        double fi = f(x, yi);
        double fi1 = f(x, yi1);
        double fm = f(x, ym);
        
        // Calculate cosine values directly
        double cos_yi = cos(ky * yi);
        double cos_yi1 = cos(ky * yi1);
        double cos_ym = cos(ky * ym);

        // Simpson's rule with cosine modulation
        sum_cos += (h / 6.0) * (
            fi * cos_yi +
            4.0 * fm * cos_ym +
            fi1 * cos_yi1
        );
    }
    
    return sum_cos;
}

// Initialize thread-local state
void init_thread_local_state() {
    thread_state.depth = 0;
}

// Adaptive subdivision for a segment based on oscillation strength
void adaptive_segment_integration(double xi, double xi1, double c, int base_Ny, 
                                 double kx, double ky, double *result) {
    double xm = (xi + xi1) / 2.0;
    
    // Calculate oscillation strength at endpoints and midpoint
    double osc_i = local_oscillation_strength(xi, c/2.0, kx, ky);
    double osc_i1 = local_oscillation_strength(xi1, c/2.0, kx, ky);
    double osc_m = local_oscillation_strength(xm, c/2.0, kx, ky);
    
    // Threshold for subdivision - adjust based on problem characteristics
    double oscil_threshold = 10.0;
    double h = xi1 - xi;
    
    // Maximum recursion depth to prevent infinite subdivision
    if (thread_state.depth > 12) {
        // Fall back to Simpson's rule if max depth reached
        double Ji = j0(kx / xi);
        double Ji1 = j0(kx / xi1);
        double Jm = j0(kx / xm);
        
        double fi = filon_inner_integral(xi, c, base_Ny, ky) * Ji;
        double fi1 = filon_inner_integral(xi1, c, base_Ny, ky) * Ji1;
        double fm = filon_inner_integral(xm, c, base_Ny, ky) * Jm;
        
        *result += (h / 6.0) * (fi + 4.0 * fm + fi1);
        return;
    }
    
    // Decide whether to subdivide based on oscillation strength
    if ((osc_i > oscil_threshold || osc_i1 > oscil_threshold || osc_m > oscil_threshold) && h > 1e-6) {
        thread_state.depth++;
        // Recursively subdivide
        adaptive_segment_integration(xi, xm, c, base_Ny, kx, ky, result);
        adaptive_segment_integration(xm, xi1, c, base_Ny, kx, ky, result);
        thread_state.depth--;
    } else {
        // Use Simpson's rule for smooth enough segments
        double Ji = j0(kx / xi);
        double Ji1 = j0(kx / xi1);
        double Jm = j0(kx / xm);
        
        double fi = filon_inner_integral(xi, c, base_Ny, ky) * Ji;
        double fi1 = filon_inner_integral(xi1, c, base_Ny, ky) * Ji1;
        double fm = filon_inner_integral(xm, c, base_Ny, ky) * Jm;
        
        *result += (h / 6.0) * (fi + 4.0 * fm + fi1);
    }
}

// Outer integral (over x) with adaptive discretization
double double_integral(double b, double c, int base_Nx, int base_Ny, double kx, double ky) {
    double integral = 0.0;
    double eps = 1e-8; // Small offset to avoid division by zero
    
    // Divide domain into initial segments
    int num_segments = 8; // Start with a small number of segments
    double h_initial = b / num_segments;
    
    #pragma omp parallel
    {
        init_thread_local_state();
        
        double local_integral = 0.0;
        int i;
        #pragma omp for nowait
        for (i = 0; i < num_segments; i++) {
            double xi = eps + i * h_initial;
            double xi1 = eps + (i + 1) * h_initial;
            
            // Apply adaptive integration to each segment
            adaptive_segment_integration(xi, xi1, c, base_Ny, kx, ky, &local_integral);
        }
        
        #pragma omp critical
        {
            integral += local_integral;
        }
    }
    
    return integral;
}

/**
 *    Initialization of the dipolar potential.
 *    kx  - array with the space mesh values in the x-direction in the K-space
 *    ky  - array with the space mesh values in the y-direction in the K-space
 *    kz  - array with the space mesh values in the z-direction in the K-space
 *    kx2 - array with the squared space mesh values in the x-direction in the
 *          K-space
 *    ky2 - array with the squared space mesh values in the y-direction in the
 *          K-space
 *    kz2 - array with the squared space mesh values in the z-direction in the
 *          K-space
 */
void initpotdd(double *kx, double *ky, double *kz, double *kx2, double *ky2, double *kz2) {
   long cnti, cntj, cntk, cntrho;
   double xk,tmp;
   double dkx, dky, dkz, krho, ctheta, stheta, dkyt;
   //FILE *cutout;

   dkx = 2. * pi / (Nx * dx);
   dky = 2. * pi / (Ny * dy);
   dkyt = dky / (sqrt(2.) - 1.0e-6);
   dkz = 2. * pi / (Nz * dz);

   for (cnti = 0; cnti < Nx2; cnti ++) kx[cnti] = cnti * dkx;
   for (cnti = Nx2; cnti < Nx; cnti ++) kx[cnti] = (cnti - Nx) * dkx;
   for (cntj = 0; cntj < Ny2; cntj ++) ky[cntj] = cntj * dky;
   for (cntj = Ny2; cntj < Ny; cntj ++) ky[cntj] = (cntj - Ny) * dky;
   for (cntk = 0; cntk < Nz2; cntk ++) kz[cntk] = cntk * dkz;
   for (cntk = Nz2; cntk < Nz; cntk ++) kz[cntk] = (cntk - Nz) * dkz;

   for (cnti = 0; cnti < Nx; cnti ++) kx2[cnti] = kx[cnti] * kx[cnti];
   for (cntj = 0; cntj < localNy; cntj ++) ky2[cntj] = ky[offsetNy + cntj] * ky[offsetNy + cntj];
   for (cntk = 0; cntk < Nz; cntk ++) kz2[cntk] = kz[cntk] * kz[cntk];

//    // //   if(rank == 0) cutout = fopen("cutoff.txt", "w");
    // for (cnti = 0; cnti < Nx; cnti ++) {
       // for (cntj = 0; cntj <= Ny; cntj ++) {
         //  krho = cntj * dkyt;
       //    cutpotdd[cnti][cntj] = cutpotddfun(cLfact * Nx, cRfact * Ny, cutmem1, cutmem2, krho, kx[cnti]);
// // // // //         if(rank == 0) fprintf(cutout, "%le %le %le\n", krho, kx[cnti], cutpotdd[cnti][cntj]);
// //
     //   }
   //  }
// // //   if(rank == 0) fclose(cutout);
   
   #pragma omp parallel for private(cnti, cntj, cntk, krho, cntrho, ctheta, stheta)
   //#pragma omp parallel for private(cnti, cntj, cntk, xk,tmp)
   for (cntj = 0; cntj < localNy; cntj ++) {
      for (cnti = 0; cnti < Nx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            krho = sqrt(kz2[cntk] + ky2[cntj]);
            cntrho = (long) (krho / dkyt);
            ctheta = kx[cnti] / sqrt(kx2[cnti] + ky2[cntj] + kz2[cntk]);
            stheta = krho / sqrt(kx2[cnti] + ky2[cntj] + kz2[cntk]);
            
             //Cilindricni cutoff
           // potdd[cntj][cnti][cntk] = 4. * pi * ((3. * ctheta * ctheta - 1.) / 3. + exp(- Lcut * krho) * (stheta * stheta * cos(Lcut * kx[cnti]) - stheta * ctheta * sin(Lcut * kx[cnti])) - cutpotdd[cnti][cntrho]);
            
            //Cilindricni cutoff bez integralnog dela
            //potdd[cntj][cnti][cntk] = 4. * pi * ((3. * ctheta * ctheta - 1.) / 3. + exp(- Lcut * krho) * (stheta * stheta * cos(Lcut * kx[cnti]) - stheta * ctheta * sin(Lcut * kx[cnti])));
            
            //Cisti deo
            //potdd[cntj][cnti][cntk]=4. * pi * ((3. * ctheta * ctheta - 1.) / 3.);
	         
             
            //Filon integral
           double rcut = 1/Rcut;  
            potdd[cntj][cnti][cntk] = 4. * pi * ((3. * ctheta * ctheta - 1.) / 3. + exp(- Lcut * krho) * (stheta * stheta * cos(Lcut * kx[cnti]) - stheta * ctheta * sin(Lcut * kx[cnti])) - double_integral(rcut,Lcut,1*Ny,1*Nx,krho,kx[cnti]));

            //Sferni cutoff
	      //   xk = sqrt(kz2[cntk] + kx2[cnti] + ky2[cntj]);  
         //   tmp = 1 + 1*(3. * cos(xk * cutoff) / (xk * xk * cutoff * cutoff) - 3. * sin(xk * cutoff) / (xk * xk * xk * cutoff * cutoff * cutoff));
         //   potdd[cntj][cnti][cntk] = (4. * pi * (3. * kx2[cnti] / (kx2[cnti] + ky2[cntj] + kz2[cntk]) - 1.) / 3.)*tmp;

            // if(rank==0){
            //    if(cnti==1 && cntj==1 && cntk==1){
            //       printf("kx = %.6f, ky = %.6f, kz = %.6f, krho = %.6f, DoubleInt = %.6f ",kx[1],ky[1],kz[1],krho,double_integral(rcut,Lcut,128,128,krho,kx[cnti]));
            //    }
            // }

         }
      }
   }
   if (rank == 0) {
      potdd[0][0][0] = 0.;
   }

   return;
}

double cutpotddfun(long NL, long NR, double **cutmem1, double *cutmem2, double cutkrho, double cutkz) {
   long cutLi, cutRj;
   double cutdL, cutdR, cutu, cutz;

   cutdL = Lcut / NL;
   cutdR = (1. / Rcut) / NR;

   #pragma omp parallel for private(cutLi, cutRj, cutu, cutz)
   for (cutLi = 0; cutLi < NL; cutLi ++) {
      cutmem1[cutLi][0] = 0.;
      for (cutRj = 1; cutRj < NR; cutRj ++) {
         cutu = cutRj * cutdR; 
         cutz = cutLi * cutdL;
         cutmem1[cutLi][cutRj] = cos(cutkz * cutz) * j0(cutkrho / cutu) * (1. - 2. * cutu * cutu * cutz * cutz) / pow(1. + cutu * cutu * cutz * cutz, 2.5);
      }
   }

   #pragma omp parallel for private(cutLi)
   for (cutLi = 0; cutLi < NL; cutLi ++) cutmem2[cutLi] = simpint(cutdR, cutmem1[cutLi], NR);

   return simpint(cutdL, cutmem2, NL);
}

/**
 *    Calculation of the wave function norm and normalization.
 *    norm - wave function norm
 *    psi  - array with the wave function values
 *    tmpx - temporary array
 *    tmpy - temporary array
 *    tmpz - temporary array
 */
void calcnorm(double *norm, double complex ***psi, double **tmpx, double **tmpy, double **tmpz) {
   int threadid, quick;
   long cnti, cntj, cntk;
   double tmp, alpha;
   void *sendbuf;

   #pragma omp parallel private(threadid, cnti, cntj, cntk, tmp)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmp = cabs(psi[cnti][cntj][cntk]);
               tmpz[threadid][cntk] = tmp * tmp;
            }
            tmpy[threadid][cntj] = simpint(dz, tmpz[threadid], Nz);
         }
         (*tmpx)[cnti] = simpint(dy, tmpy[threadid], Ny);
      }
   }

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      norm_psi2 = simpint(dx, *tmpx, Nx);
      alpha = 1. / norm_psi2;
      *norm = sqrt(alpha);
      norm_psi3 = 0.;
   }

   if (QDEPL == 1) {
   #pragma omp parallel private(threadid, cnti, cntj, cntk, tmp)
   {
       threadid = omp_get_thread_num();

       #pragma omp for
       for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmp = cabs(psi[cnti][cntj][cntk]);
               tmpz[threadid][cntk] = tmp * tmp * tmp;
            }
            tmpy[threadid][cntj] = simpint(dz, tmpz[threadid], Nz);
         }
         (*tmpx)[cnti] = simpint(dy, tmpy[threadid], Ny);
      }
   }

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      norm_psi3 = simpint(dx, *tmpx, Nx);

      for (quick = 0; quick < 50; quick ++) {
         alpha = (1. - pow(alpha, 1.5) * h4 * norm_psi3) / norm_psi2;
      }

      *norm = sqrt(alpha);
   }
   }

   MPI_Bcast(norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&norm_psi2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&norm_psi3, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (optimre == 2) {
      kN = norm_psi2 + h4 * norm_psi3;
      Nad *= kN;
      g *= kN;
      gd *= kN;
      g3 *= kN * kN;
      h2 *= pow(kN, 1.5);
      h4 *= sqrt(kN);
   }

   norm_psi2 *= *norm * *norm;
   norm_psi3 *= *norm * *norm * *norm;

   #pragma omp for private(cnti, cntj, cntk)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            psi[cnti][cntj][cntk] *= *norm;
         }
      }
   }

   return;
}

/**
 *    Calculation of the chemical potential and energy.
 *    mu        - chemical potential
 *    en        - energy
 *    psi       - array with the wave function values
 *    psi_t       - array with the transposed wave function values
 *    dpsi      - temporary array
 *    dpsi_t    - temporary array
 *    psidd2    - array with the squared wave function values
 *    psidd2fft - array with the squared wave function fft values
 *    tmpxi     - temporary array
 *    tmpyi     - temporary array
 *    tmpzi     - temporary array
 *    tmpxj     - temporary array
 *    tmpyj     - temporary array
 *    tmpzj     - temporary array
 */
void calcmudet(double *mu, double complex ***psi, double complex ***psi_t, double complex ***dpsi, double ***dpsiL, double complex ***dpsi_t, double ***psidd2, double ***psidd20, fftw_complex *psidd2fft, double **tmpxi, double **tmpyi, double **tmpzi, double **tmpxj, double **tmpyj, double **tmpzj, double complex **tmpdpsi1, double complex **tmpdpsi2, double ***tmpxmui, double ***tmpymui, double ***tmpzmui, double *Lx, double complex ***Gpsi, double complex ***Gpsi_t) {
   int threadid;
   long cnti, cntj, cntk, cnte;
   double psi2, psi3, tmp, ap;
   double complex p, pd, cp, cpd, pdd, cpdd;
   void *sendbuf;

   calcpsidd2(psi, psidd2, psidd2fft);
   calcpsidd20(psi, psidd20, psidd2fft);

   //fftw_execute(plan_transpose_x);

   #pragma omp parallel private(threadid, cnti, cntj, cntk, p, pd, cp, ap, cpd, pdd, cpdd)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cntj = 0; cntj < localNy; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cnti = 0; cnti < Nx; cnti ++) {
               tmpdpsi2[threadid][cnti] = psi_t[cnti][cntj][cntk];
            }
            diffc(dx, tmpdpsi2[threadid], tmpdpsi1[threadid], Nx);
            diffc(dx, tmpdpsi1[threadid], tmpdpsi2[threadid], Nx);
            for (cnti = 0; cnti < Nx; cnti ++) {
               p = psi_t[cnti][cntj][cntk];
               pd = tmpdpsi1[threadid][cnti];
               cp = conj(p);
               ap = cabs(p);
               cpd = conj(pd);
               pdd = tmpdpsi2[threadid][cnti];
               cpdd = conj(pdd);
               dpsi_t[cnti][cntj][cntk] = pd * cpd;
               Gpsi_t[cnti][cntj][cntk] = CC[1] * cp * pd * pd / ((p + EPSILON) * (ap + EPSILON)) +
                                          CC[4] * p * p * cpd * cpd / ((ap + EPSILON) * (ap + EPSILON) * (ap + EPSILON)) +
                                          CC[7] * pd * cpd / (ap + EPSILON) + CC[10] * ap * pdd / (p + EPSILON) +
                                          CC[13] * p * cpdd / (ap + EPSILON);
            }
         }
      }
   }

   fftw_execute(plan_transpose_dpsi);
   fftw_execute(plan_transpose_Gpsi);

   #pragma omp parallel private(threadid, cnti, cntj, cntk, cnte, psi2, psi3, tmp, p, pd, cp, ap, cpd, pdd, cpdd)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cntj = 0; cntj < Ny; cntj ++) {
               tmpdpsi2[threadid][cntj] = psi[cnti][cntj][cntk];
            }
            diffc(dy, tmpdpsi2[threadid], tmpdpsi1[threadid], Ny);
            diffc(dy, tmpdpsi1[threadid], tmpdpsi2[threadid], Ny);
            for (cntj = 0; cntj < Ny; cntj ++) {
               p = psi[cnti][cntj][cntk];
               pd = tmpdpsi1[threadid][cntj];
               cp = conj(p);
               ap = cabs(p);
               cpd = conj(pd);
               pdd = tmpdpsi2[threadid][cntj];
               cpdd = conj(pdd);
               dpsi[cnti][cntj][cntk] += pd * cpd;
               dpsiL[cnti][cntj][cntk] = 2. * cimag(p) * z[cntk] * creal(pd);
               Gpsi[cnti][cntj][cntk] += CC[2] * cp * pd * pd / ((p + EPSILON) * (ap + EPSILON)) +
                                         CC[5] * p * p * cpd * cpd / ((ap + EPSILON) * (ap + EPSILON) * (ap + EPSILON)) +
                                         CC[8] * pd * cpd / (ap + EPSILON) + CC[11] * ap * pdd / (p + EPSILON) +
                                         CC[14] * p * cpdd / (ap + EPSILON);
            }
         }
      }
      #pragma omp barrier

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmpdpsi2[threadid][cntk] = psi[cnti][cntj][cntk];
            }
            diffc(dz, tmpdpsi2[threadid], tmpdpsi1[threadid], Nz);
            diffc(dz, tmpdpsi1[threadid], tmpdpsi2[threadid], Nz);
            for (cntk = 0; cntk < Nz; cntk ++) {
               p = psi[cnti][cntj][cntk];
               pd = tmpdpsi1[threadid][cntk];
               cp = conj(p);
               ap = cabs(p);
               cpd = conj(pd);
               pdd = tmpdpsi2[threadid][cntk];
               cpdd = conj(pdd);
               dpsi[cnti][cntj][cntk] += pd * cpd;
               dpsiL[cnti][cntj][cntk] += - 2. * cimag(p) * y[cntj] * creal(pd);
               Gpsi[cnti][cntj][cntk] += CC[3] * cp * pd * pd / ((p + EPSILON) * (ap + EPSILON)) +
                                         CC[6] * p * p * cpd * cpd / ((ap + EPSILON) * (ap + EPSILON) * (ap + EPSILON)) +
                                         CC[9] * pd * cpd / (ap + EPSILON) + CC[12] * ap * pdd / (p + EPSILON) +
                                         CC[15] * p * cpdd / (ap + EPSILON);
            }
         }
      }
      #pragma omp barrier

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmp = cabs(psi[cnti][cntj][cntk]);
               psi2 = tmp * tmp;
               psi3 = psi2 * tmp;
               tmpzmui[0][threadid][cntk] = creal(dpsi[cnti][cntj][cntk]);
               tmpzmui[1][threadid][cntk] = pot[cnti][cntj][cntk] * psi2;
               tmpzmui[2][threadid][cntk] = g * psi2 * psi2;
               //tmpzmui[2][threadid][cntk] = 1 * psi2 * psi2;
               tmpzmui[3][threadid][cntk] = gd * psidd20[cnti][cntj][cntk] * psi2;
               tmpzmui[4][threadid][cntk] = h2 * psi3 * psi2;
               tmpzmui[5][threadid][cntk] = gd * (psidd2[cnti][cntj][cntk] - psidd20[cnti][cntj][cntk]) * psi2;
               tmpzmui[6][threadid][cntk] = creal(Gpsi[cnti][cntj][cntk]) * psi2;
               tmpzmui[7][threadid][cntk] = dpsiL[cnti][cntj][cntk];
            }
            for (cnte = 0; cnte < 8; cnte ++) tmpymui[cnte][threadid][cntj] = simpint(dz, tmpzmui[cnte][threadid], Nz);
         }
         for (cnte = 0; cnte < 8; cnte ++) tmpxmui[cnte][0][cnti] = simpint(dy, tmpymui[cnte][threadid], Ny);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   for (cnte = 0; cnte < 8; cnte ++) {
      sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpxmui[cnte][0];
      MPI_Gather(sendbuf, localNx, MPI_DOUBLE, tmpxmui[cnte][0], localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }

   if (rank == 0) {
      for (cnte = 0; cnte < 8; cnte ++) mu[cnte] = simpint(dx, tmpxmui[cnte][0], Nx);
      *Lx = mu[7];
      mu[7] *= - omega;
      //printf("Mu = %.26f, g=%.26f \n",mu[2], g);
   }

   return;
}

void calcmu(double *mu, double complex ***psi, double complex ***psi_t, double complex ***dpsi, double ***dpsiL, double complex ***dpsi_t, double ***psidd2, fftw_complex *psidd2fft, double **tmpxi, double **tmpyi, double **tmpzi, double **tmpxj, double **tmpyj, double **tmpzj, double complex **tmpdpsi1, double complex **tmpdpsi2, double complex ***Gpsi, double complex ***Gpsi_t) {
   int threadid;
   long cnti, cntj, cntk;
   double dpsi2, psi2, psi3, psi2lin, psidd2lin, tmp, ap;
   double complex p, pd, cp, cpd, pdd, cpdd;
   void *sendbuf;

   calcpsidd2(psi, psidd2, psidd2fft);

   //fftw_execute(plan_transpose_x);

   #pragma omp parallel private(threadid, cnti, cntj, cntk, p, pd, cp, ap, cpd, pdd, cpdd)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cntj = 0; cntj < localNy; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cnti = 0; cnti < Nx; cnti ++) {
               tmpdpsi2[threadid][cnti] = psi_t[cnti][cntj][cntk];
            }
            diffc(dx, tmpdpsi2[threadid], tmpdpsi1[threadid], Nx);
            diffc(dx, tmpdpsi1[threadid], tmpdpsi2[threadid], Nx);
            for (cnti = 0; cnti < Nx; cnti ++) {
               p = psi_t[cnti][cntj][cntk];
               pd = tmpdpsi1[threadid][cnti];
               cp = conj(p);
               ap = cabs(p);
               cpd = conj(pd);
               pdd = tmpdpsi2[threadid][cnti];
               cpdd = conj(pdd);
               dpsi_t[cnti][cntj][cntk] = pd * cpd;
               Gpsi_t[cnti][cntj][cntk] = CC[1] * cp * pd * pd / ((p + EPSILON) * (ap + EPSILON)) +
                                          CC[4] * p * p * cpd * cpd / ((ap + EPSILON) * (ap + EPSILON) * (ap + EPSILON)) +
                                          CC[7] * pd * cpd / (ap + EPSILON) + CC[10] * ap * pdd / (p + EPSILON) +
                                          CC[13] * p * cpdd / (ap + EPSILON);
            }
         }
      }
   }

   fftw_execute(plan_transpose_dpsi);
   fftw_execute(plan_transpose_Gpsi);

   #pragma omp parallel private(threadid, cnti, cntj, cntk, p, pd, cp, ap, cpd, pdd, cpdd,  dpsi2, psi2, psi3, psi2lin, psidd2lin, tmp)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cntj = 0; cntj < Ny; cntj ++) {
               tmpdpsi2[threadid][cntj] = psi[cnti][cntj][cntk];
            }
            diffc(dy, tmpdpsi2[threadid], tmpdpsi1[threadid], Ny);
            diffc(dy, tmpdpsi1[threadid], tmpdpsi2[threadid], Ny);
            for (cntj = 0; cntj < Ny; cntj ++) {
               p = psi[cnti][cntj][cntk];
               pd = tmpdpsi1[threadid][cntj];
               cp = conj(p);
               ap = cabs(p);
               cpd = conj(pd);
               pdd = tmpdpsi2[threadid][cntj];
               cpdd = conj(pdd);
               dpsi[cnti][cntj][cntk] += pd * cpd;
               dpsiL[cnti][cntj][cntk] = 2. * cimag(p) * z[cntk] * creal(pd);
               Gpsi[cnti][cntj][cntk] += CC[2] * cp * pd * pd / ((p + EPSILON) * (ap + EPSILON)) +
                                         CC[5] * p * p * cpd * cpd / ((ap + EPSILON) * (ap + EPSILON) * (ap + EPSILON)) +
                                         CC[8] * pd * cpd / (ap + EPSILON) + CC[11] * ap * pdd / (p + EPSILON) +
                                         CC[14] * p * cpdd / (ap + EPSILON);
            }
         }
      }
      #pragma omp barrier

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmpdpsi2[threadid][cntk] = psi[cnti][cntj][cntk];
            }
            diffc(dz, tmpdpsi2[threadid], tmpdpsi1[threadid], Nz);
            diffc(dz, tmpdpsi1[threadid], tmpdpsi2[threadid], Nz);
            for (cntk = 0; cntk < Nz; cntk ++) {
               p = psi[cnti][cntj][cntk];
               pd = tmpdpsi1[threadid][cntk];
               cp = conj(p);
               ap = cabs(p);
               cpd = conj(pd);
               pdd = tmpdpsi2[threadid][cntk];
               cpdd = conj(pdd);
               dpsi[cnti][cntj][cntk] += pd * cpd;
               dpsiL[cnti][cntj][cntk] += - 2. * cimag(p) * y[cntj] * creal(pd);
               Gpsi[cnti][cntj][cntk] += CC[3] * cp * pd * pd / ((p + EPSILON) * (ap + EPSILON)) +
                                         CC[6] * p * p * cpd * cpd / ((ap + EPSILON) * (ap + EPSILON) * (ap + EPSILON)) +
                                         CC[9] * pd * cpd / (ap + EPSILON) + CC[12] * ap * pdd / (p + EPSILON) +
                                         CC[15] * p * cpdd / (ap + EPSILON);
            }
         }
      }
      #pragma omp barrier
   
      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               dpsi2 = creal(dpsi[cnti][cntj][cntk]);
               tmp = cabs(psi[cnti][cntj][cntk]);
               psi2 = tmp * tmp;
               psi3 = psi2 * tmp;
               psi2lin = g * psi2;
               psidd2lin = gd * psidd2[cnti][cntj][cntk];
               tmpzi[threadid][cntk] = dpsi2 + (pot[cnti][cntj][cntk] + psi2lin + psidd2lin + h2 * psi3) * psi2 + creal(Gpsi[cnti][cntj][cntk]) * psi2 - omega * dpsiL[cnti][cntj][cntk];
            }
            tmpyi[threadid][cntj] = simpint(dz, tmpzi[threadid], Nz);
         }
         (*tmpxi)[cnti] = simpint(dy, tmpyi[threadid], Ny);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpxi;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpxi, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) *mu = simpint(dx, *tmpxi, Nx);

   return;
}

/**
 *    Calculation of squared wave function values for dipole-dipole
 *    interaction.
 *    psi       - array with the wave function values
 *    psidd2    - array with the squared wave function values
 *    psidd2fft - array with the squared wave function fft values
 */
void calcpsidd2(double complex ***psi, double ***psidd2, fftw_complex *psidd2fft) {
   long cnti, cntj, cntk;
   long last = 0;
   double *psidd2tmp = (double *) psidd2fft;
   double tmp;

   #pragma omp parallel for private(cnti, cntj, cntk, tmp)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmp = cabs(psi[cnti][cntj][cntk]);
            psidd2[cnti][cntj][cntk] = tmp * tmp * (1. + h4 * tmp);
         }
      }
   }

   fftw_execute(plan_forward);

   #pragma omp parallel for private(cnti, cntj, cntk)
   for (cntj = 0; cntj < localNy; cntj ++) {
      for (cnti = 0; cnti < Nx; cnti ++) {
         for (cntk = 0; cntk < Nz2 + 1; cntk ++) {
            psidd2fft[cntj * Nx * (Nz2 + 1) + cnti * (Nz2 + 1) + cntk][0] *= potdd[cntj][cnti][cntk];
            psidd2fft[cntj * Nx * (Nz2 + 1) + cnti * (Nz2 + 1) + cntk][1] *= potdd[cntj][cnti][cntk];
         }
      }
   }

   fftw_execute(plan_backward);

   #pragma omp parallel for private(cnti, cntj, cntk)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            psidd2[cnti][cntj][cntk] /= (Nx * Ny * Nz);
         }
      }
   }

   if (nprocs > 1) {
      if (rank == 0) {
         MPI_Send(psidd2[0][0], Ny * Nz, MPI_DOUBLE, nprocs - 1, 0, MPI_COMM_WORLD);
      } else if (rank == nprocs - 1) {
         MPI_Recv(psidd2tmp, Ny * Nz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         last = 1;
      }
   } else {
      psidd2tmp = psidd2[0][0];
      last = 1;
   }

   if (rank == nprocs - 1) {
      #pragma omp parallel for private(cntj, cntk)
      for (cntj = 0; cntj < Ny - 1; cntj ++) {
         for (cntk = 0; cntk < Nz - 1; cntk ++) {
            psidd2[localNx - 1][cntj][cntk] = psidd2tmp[cntj * Nz + cntk];
         }
      }
   }

   #pragma omp parallel for private(cnti, cntk)
   for (cnti = 0; cnti < localNx - last; cnti ++) {
      for (cntk = 0; cntk < Nz - 1; cntk ++) {
         psidd2[cnti][Ny - 1][cntk] = psidd2[cnti][0][cntk];
      }
   }

   #pragma omp parallel for private(cnti, cntj)
   for (cnti = 0; cnti < localNx - last; cnti ++) {
      for (cntj = 0; cntj < Ny - 1; cntj ++) {
         psidd2[cnti][cntj][Nz - 1] = psidd2[cnti][cntj][0];
      }
   }

   return;
}

void calcpsidd20(double complex ***psi, double ***psidd20, fftw_complex *psidd2fft) {            
   long cnti, cntj, cntk;
   long last = 0;
   double *psidd2tmp = (double *) psidd2fft;
   double tmp;

   #pragma omp parallel for private(cnti, cntj, cntk, tmp)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmp = cabs(psi[cnti][cntj][cntk]);
            psidd20[cnti][cntj][cntk] = tmp * tmp;
         }
      }
   }

   fftw_execute(plan_forward0);

   #pragma omp parallel for private(cnti, cntj, cntk)
   for (cntj = 0; cntj < localNy; cntj ++) {
      for (cnti = 0; cnti < Nx; cnti ++) {
         for (cntk = 0; cntk < Nz2 + 1; cntk ++) {
            psidd2fft[cntj * Nx * (Nz2 + 1) + cnti * (Nz2 + 1) + cntk][0] *= potdd[cntj][cnti][cntk];
            psidd2fft[cntj * Nx * (Nz2 + 1) + cnti * (Nz2 + 1) + cntk][1] *= potdd[cntj][cnti][cntk];
         }
      }
   }

   fftw_execute(plan_backward0);

   #pragma omp parallel for private(cnti, cntj, cntk)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            psidd20[cnti][cntj][cntk] /= (Nx * Ny * Nz);
         }
      }
   }

   if (nprocs > 1) {
      if (rank == 0) {
         MPI_Send(psidd20[0][0], Ny * Nz, MPI_DOUBLE, nprocs - 1, 0, MPI_COMM_WORLD);
      } else if (rank == nprocs - 1) {
         MPI_Recv(psidd2tmp, Ny * Nz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         last = 1;
      }
   } else {
      psidd2tmp = psidd20[0][0];
      last = 1;
   }

   if (rank == nprocs - 1) {
      #pragma omp parallel for private(cntj, cntk)
      for (cntj = 0; cntj < Ny - 1; cntj ++) {
         for (cntk = 0; cntk < Nz - 1; cntk ++) {
            psidd20[localNx - 1][cntj][cntk] = psidd2tmp[cntj * Nz + cntk];
         }
      }
   }

   #pragma omp parallel for private(cnti, cntk)
   for (cnti = 0; cnti < localNx - last; cnti ++) {
      for (cntk = 0; cntk < Nz - 1; cntk ++) {
         psidd20[cnti][Ny - 1][cntk] = psidd20[cnti][0][cntk];
      }
   }

   #pragma omp parallel for private(cnti, cntj)
   for (cnti = 0; cnti < localNx - last; cnti ++) {
      for (cntj = 0; cntj < Ny - 1; cntj ++) {
         psidd20[cnti][cntj][Nz - 1] = psidd20[cnti][cntj][0];
      }
   }

   return;
}

/**
 *    Calculation of the root mean square radius.
 *    rms   - root mean square radius
 *    psi   - array with the wave function values
 *    psi_t - array with the transposed wave function values
 *    tmpx  - temporary array
 *    tmpy  - temporary array
 *    tmpz  - temporary array
 */
void calcrms(double *rms, double complex ***psi, double complex ***psi_t, double **tmpx, double **tmpy, double **tmpz) {
   int threadid;
   long cnti, cntj, cntk;
   double psi2;
   void *sendbuf;

   fftw_execute(plan_transpose_x);

   #pragma omp parallel private(threadid, cnti, cntj, cntk, psi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cntj = 0; cntj < localNy; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cnti = 0; cnti < Nx; cnti ++) {
               psi2 = cabs(psi_t[cnti][cntj][cntk]);
               psi2 *= psi2;
               tmpx[threadid][cnti] = x2[cnti] * psi2;
            }
            tmpz[threadid][cntk] = simpint(dx, tmpx[threadid], Nx);
         }
         (*tmpy)[cntj] = simpint(dz, tmpz[threadid], Nz);
      }
   }

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpy;
   MPI_Gather(sendbuf, localNy, MPI_DOUBLE, *tmpy, localNy, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      rms[1] = sqrt(simpint(dy, *tmpy, Ny));
   }

   #pragma omp parallel private(threadid, cnti, cntj, cntk, psi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cntj = 0; cntj < Ny; cntj ++) {
               psi2 = cabs(psi[cnti][cntj][cntk]);
               psi2 *= psi2;
               tmpy[threadid][cntj] = y2[cntj] * psi2;
            }
            tmpz[threadid][cntk] = simpint(dy, tmpy[threadid], Ny);
         }
         (*tmpx)[cnti] = simpint(dz, tmpz[threadid], Nz);
      }
   }

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      rms[2] = sqrt(simpint(dx, *tmpx, Nx));
   }

   #pragma omp parallel private(threadid, cnti, cntj, cntk, psi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               psi2 = cabs(psi[cnti][cntj][cntk]);
               psi2 *= psi2;
               tmpz[threadid][cntk] = z2[cntk] * psi2;
            }
            tmpy[threadid][cntj] = simpint(dz, tmpz[threadid], Nz);
         }
         (*tmpx)[cnti] = simpint(dy, tmpy[threadid], Ny);
      }
   }

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      rms[3] = sqrt(simpint(dx, *tmpx, Nx));
      rms[0] = sqrt(rms[1] * rms[1] + rms[2] * rms[2] + rms[3] * rms[3]);
   }

   return;
}
/**
 *    Time propagation with respect to H1 (part of the Hamiltonian without
 *    spatial derivatives).
 *    psi       - array with the wave function values
 *    psidd2    - array with the squared wave function values
 *    psidd2fft - array with the squared wave function fft values
 */
void calcnu(double complex ***psi, double complex ***psi_t, double complex ***Gpsi, double complex ***Gpsi_t, double ***psidd2, fftw_complex *psidd2fft, double complex **tmpdpsi1, double complex **tmpdpsi2) {
   long cnti, cntj, cntk;
   double psi2, psi3, psi2lin, psidd2lin, ap;
   double complex p, pd, cp, cpd, pdd, cpdd, tmp;
   int threadid;

   calcpsidd2(psi, psidd2, psidd2fft);

   fftw_execute(plan_transpose_x);

   #pragma omp parallel private(threadid, cnti, cntj, cntk, p, pd, cp, ap, cpd, pdd, cpdd)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cntj = 0; cntj < localNy; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cnti = 0; cnti < Nx; cnti ++) {
               tmpdpsi2[threadid][cnti] = psi_t[cnti][cntj][cntk];
            }
            diffc(dx, tmpdpsi2[threadid], tmpdpsi1[threadid], Nx);
            diffc(dx, tmpdpsi1[threadid], tmpdpsi2[threadid], Nx);
            for (cnti = 0; cnti < Nx; cnti ++) {
               p = psi_t[cnti][cntj][cntk];
               pd = tmpdpsi1[threadid][cnti];
               cp = conj(p);
               ap = cabs(p);
               cpd = conj(pd);
               pdd = tmpdpsi2[threadid][cnti];
               cpdd = conj(pdd);
               Gpsi_t[cnti][cntj][cntk] = CC[1] * cp * pd * pd / ((p + EPSILON) * (ap + EPSILON)) +
                                          CC[4] * p * p * cpd * cpd / ((ap + EPSILON) * (ap + EPSILON) * (ap + EPSILON)) +
                                          CC[7] * pd * cpd / (ap + EPSILON) + CC[10] * ap * pdd / (p + EPSILON) +
                                          CC[13] * p * cpdd / (ap + EPSILON);
            }
         }
      }
   }
   #pragma omp barrier

   fftw_execute(plan_transpose_Gpsi);

   #pragma omp parallel private(threadid, cnti, cntj, cntk, p, pd, cp, ap, cpd, pdd, cpdd)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cntj = 0; cntj < Ny; cntj ++) {
               tmpdpsi2[threadid][cntj] = psi[cnti][cntj][cntk];
            }
            diffc(dy, tmpdpsi2[threadid], tmpdpsi1[threadid], Ny);
            diffc(dy, tmpdpsi1[threadid], tmpdpsi2[threadid], Ny);
            for (cntj = 0; cntj < Ny; cntj ++) {
               p = psi[cnti][cntj][cntk];
               pd = tmpdpsi1[threadid][cntj];
               cp = conj(p);
               ap = cabs(p);
               cpd = conj(pd);
               pdd = tmpdpsi2[threadid][cntj];
               cpdd = conj(pdd);
               Gpsi[cnti][cntj][cntk] += CC[2] * cp * pd * pd / ((p + EPSILON) * (ap + EPSILON)) +
                                         CC[5] * p * p * cpd * cpd / ((ap + EPSILON) * (ap + EPSILON) * (ap + EPSILON)) +
                                         CC[8] * pd * cpd / (ap + EPSILON) + CC[11] * ap * pdd / (p + EPSILON) +
                                         CC[14] * p * cpdd / (ap + EPSILON);
            }
         }
      }
   }
   #pragma omp barrier

   #pragma omp parallel private(threadid, cnti, cntj, cntk, p, pd, cp, ap, cpd, pdd, cpdd)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmpdpsi2[threadid][cntk] = psi[cnti][cntj][cntk];
            }
            diffc(dz, tmpdpsi2[threadid], tmpdpsi1[threadid], Nz);
            diffc(dz, tmpdpsi1[threadid], tmpdpsi2[threadid], Nz);
            for (cntk = 0; cntk < Nz; cntk ++) {
               p = psi[cnti][cntj][cntk];
               pd = tmpdpsi1[threadid][cntk];
               cp = conj(p);
               ap = cabs(p);
               cpd = conj(pd);
               pdd = tmpdpsi2[threadid][cntk];
               cpdd = conj(pdd);
               Gpsi[cnti][cntj][cntk] += CC[3] * cp * pd * pd / ((p + EPSILON) * (ap + EPSILON)) +
                                         CC[6] * p * p * cpd * cpd / ((ap + EPSILON) * (ap + EPSILON) * (ap + EPSILON)) +
                                         CC[9] * pd * cpd / (ap + EPSILON) + CC[12] * ap * pdd / (p + EPSILON) +
                                         CC[15] * p * cpdd / (ap + EPSILON);
            }
         }
      }
   }
   #pragma omp barrier

   #pragma omp parallel for private(cnti, cntj, cntk, psi2, psi3, psi2lin, psidd2lin, tmp)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmp = cabs(psi[cnti][cntj][cntk]);
            psi2 = tmp * tmp;
            psi3 = psi2 * tmp;
            psi2lin = g * psi2;
            psidd2lin = gd * psidd2[cnti][cntj][cntk];
            tmp = dt * (pot[cnti][cntj][cntk] + psi2lin + psidd2lin + h2 * psi3 + Gpsi[cnti][cntj][cntk]);
            psi[cnti][cntj][cntk] *= (*e)(- tmp) * exp(- dt * g3 * psi2 * psi2);
         }
      }
   }

   return;
}

/**
 *    Time propagation with respect to H2 (x-part of the Laplacian).
 *    psi_t - array with the wave function values (transposed)
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calclux(double complex ***psi, double complex ***psi_t, double complex **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double complex c;

   fftw_execute(plan_transpose_x);

   #pragma omp parallel private(threadid, cnti, cntj, cntk, c)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cntj = 0; cntj < localNy; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            cbeta[threadid][Nx - 2] = 0.;
            for (cnti = Nx - 2; cnti > 0; cnti --) {
               c = - Ax * psi_t[cnti + 1][cntj][cntk] + Ax0r * psi_t[cnti][cntj][cntk] - Ax * psi_t[cnti - 1][cntj][cntk];
               cbeta[threadid][cnti - 1] =  cgammax[cnti] * (Ax * cbeta[threadid][cnti] - c);
            }
            psi_t[0][cntj][cntk] = 0.;
            for (cnti = 0; cnti < Nx - 2; cnti ++) {
               psi_t[cnti + 1][cntj][cntk] = calphax[cnti] * psi_t[cnti][cntj][cntk] + cbeta[threadid][cnti];
            }
            psi_t[Nx - 1][cntj][cntk] = 0.;
         }
      }
   }

   fftw_execute(plan_transpose_y);

   return;
}

/**
 *    Time propagation with respect to H3 (y-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calcluy(double complex ***psi, double complex **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double complex c;

   #pragma omp parallel private(threadid, cnti, cntj, cntk, c)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            cbeta[threadid][Ny - 2] = 0.;
            for (cntj = Ny - 2; cntj > 0; cntj --) {
               c = - Ayp[cntk] * psi[cnti][cntj + 1][cntk] + Ay0r * psi[cnti][cntj][cntk] - Aym[cntk] * psi[cnti][cntj - 1][cntk];
               cbeta[threadid][cntj - 1] =  cgammay[cntk][cntj] * (Ayp[cntk] * cbeta[threadid][cntj] - c);
            }
            psi[cnti][0][cntk] = 0.;
            for (cntj = 0; cntj < Ny - 2; cntj ++) {
               psi[cnti][cntj + 1][cntk] = calphay[cntk][cntj] * psi[cnti][cntj][cntk] + cbeta[threadid][cntj];
            }
            psi[cnti][Ny - 1][cntk] = 0.;
         }
      }
   }

   return;
}

/**
 *    Time propagation with respect to H4 (z-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calcluz(double complex ***psi, double complex **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double complex c;

   #pragma omp parallel private(threadid, cnti, cntj, cntk, c)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            cbeta[threadid][Nz - 2] = 0.;
            for (cntk = Nz - 2; cntk > 0; cntk --) {
               c = - Azp[cntj] * psi[cnti][cntj][cntk + 1] + Az0r * psi[cnti][cntj][cntk] - Azm[cntj] * psi[cnti][cntj][cntk - 1];
               cbeta[threadid][cntk - 1] =  cgammaz[cntk][cntj] * (Azp[cntj] * cbeta[threadid][cntk] - c);
            }
            psi[cnti][cntj][0] = 0.;
            for (cntk = 0; cntk < Nz - 2; cntk ++) {
               psi[cnti][cntj][cntk + 1] = calphaz[cntk][cntj] * psi[cnti][cntj][cntk] + cbeta[threadid][cntk];
            }
            psi[cnti][cntj][Nz - 1] = 0.;
         }
      }
   }

   return;
}
void outdenx(double complex ***psi, double **outx, double *tmpy, double *tmpz, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 2 * sizeof(double) * (localNx / outstpx);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmpz[cntk] = cabs(psi[cnti][cntj][cntk]);
            tmpz[cntk] *= tmpz[cntk];
         }
         tmpy[cntj] = simpint(dz, tmpz, Nz);
      }
      outx[cnti / outstpx][0] = x[offsetNx + cnti];
      outx[cnti / outstpx][1] = simpint(dy, tmpy, Ny);
   }

   MPI_File_write_at_all(file, fileoffset, *outx, (localNx / outstpx) * 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
}

void outdeny(double complex ***psi_t, double **outy, double *tmpx, double *tmpz, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 2 * sizeof(double) * (localNy / outstpy);

   fftw_execute(plan_transpose_x);

   for (cntj = 0; cntj < localNy; cntj += outstpy) {
      for (cnti = 0; cnti < Nx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmpz[cntk] = cabs(psi_t[cnti][cntj][cntk]);
            tmpz[cntk] *= tmpz[cntk];
         }
         tmpx[cnti] = simpint(dz, tmpz, Nz);
      }
      outy[cntj / outstpy][0] = y[offsetNy + cntj];
      outy[cntj / outstpy][1] = simpint(dx, tmpx, Nx);
   }

   MPI_File_write_at_all(file, fileoffset, *outy, (localNy / outstpy) * 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
}

void outdenz(double complex ***psi, double **outz, double *tmpx, double *tmpy, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;
   void *sendbuf;

   sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpx;

   for (cntk = 0; cntk < Nz; cntk += outstpz) {
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            tmpy[cntj] = cabs(psi[cnti][cntj][cntk]);
            tmpy[cntj] *= tmpy[cntj];
         }
         tmpx[cnti] = simpint(dy, tmpy, Ny);
      }

      MPI_Gather(sendbuf, localNx, MPI_DOUBLE, tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      if (rank == 0) {
         outz[cntk / outstpz][0] = z[cntk];
         outz[cntk / outstpz][1] = simpint(dx, tmpx, Nx);
      }
   }

   if (rank == 0) {
      fileoffset = 0;
      MPI_File_write_at(file, fileoffset, *outz, (Nz / outstpz) * 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
   }
}

void outdenxy(double complex ***psi, double ***outxy, double *tmpz, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Ny / outstpy);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmpz[cntk] = cabs(psi[cnti][cntj][cntk]);
            tmpz[cntk] *= tmpz[cntk];
         }
         outxy[cnti / outstpx][cntj / outstpy][0] = x[offsetNx + cnti];
         outxy[cnti / outstpx][cntj / outstpy][1] = y[cntj];
         outxy[cnti / outstpx][cntj / outstpy][2] = simpint(dz, tmpz, Nz);
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxy, (localNx / outstpx) * (Ny / outstpy) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outdenxz(double complex ***psi, double ***outxz, double *tmpy, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Nz / outstpz);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            tmpy[cntj] = cabs(psi[cnti][cntj][cntk]);
            tmpy[cntj] *= tmpy[cntj];
         }
         outxz[cnti / outstpx][cntk / outstpz][0] = x[offsetNx + cnti];
         outxz[cnti / outstpx][cntk / outstpz][1] = z[cntk];
         outxz[cnti / outstpx][cntk / outstpz][2] = simpint(dy, tmpy, Ny);
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxz, (localNx / outstpx) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outdenyz(double complex ***psi_t, double ***outyz, double *tmpx, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNy / outstpy) * (Nz / outstpz);

   fftw_execute(plan_transpose_x);

   for (cntj = 0; cntj < localNy; cntj += outstpy) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         for (cnti = 0; cnti < Nx; cnti ++) {
            tmpx[cnti] = cabs(psi_t[cnti][cntj][cntk]);
            tmpx[cnti] *= tmpx[cnti];
         }
         outyz[cntj / outstpy][cntk / outstpz][0] = y[offsetNy + cntj];
         outyz[cntj / outstpy][cntk / outstpz][1] = z[cntk];
         outyz[cntj / outstpy][cntk / outstpz][2] = simpint(dx, tmpx, Nx);
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outyz, (localNy / outstpy) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
}

void outpsi2xy(double complex ***psi, double ***outxy, MPI_File file) {
   long cnti, cntj;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Ny / outstpy);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         outxy[cnti / outstpx][cntj / outstpy][0] = x[offsetNx + cnti];
         outxy[cnti / outstpx][cntj / outstpy][1] = y[cntj];
         outxy[cnti / outstpx][cntj / outstpy][2] = cabs(psi[cnti][cntj][Nz2]) * cabs(psi[cnti][cntj][Nz2]);
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxy, (localNx / outstpx) * (Ny / outstpy) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outpsi2x(double complex ***psi, double **outpsix, MPI_File file) {
   long cnti;
   MPI_Offset fileoffset;

   fileoffset = rank * 2 * sizeof(double) * (localNx / outstpx);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      
         outpsix[cnti / outstpx][0] = x[offsetNx + cnti];
      
         outpsix[cnti / outstpx][1] = cabs(psi[cnti][Ny2][Nz2]) * cabs(psi[cnti][Ny2][Nz2]);
         //outpsix[cnti / outstpx][1] = psi[cnti][Ny2][Nz2] ;
      
   }

   MPI_File_write_at_all(file, fileoffset, *outpsix, (localNx / outstpx) *2, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outpsi2y(double complex ***psi, double **outpsiy, MPI_File file) {
   long cntj;

   // Determine the process that owns x[Nx2]
   int global_x_index = Nx2;
   if (offsetNx <= global_x_index && global_x_index < offsetNx + localNx) {
      // Compute the local x-index corresponding to the global index
      int cnti = global_x_index - offsetNx;

      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         outpsiy[cntj / outstpy][0] = y[cntj];
         outpsiy[cntj / outstpy][1] = cabs(psi[cnti][cntj][Nz2]) * cabs(psi[cnti][cntj][Nz2]);
      }

      // Compute the file offset
      MPI_Offset fileoffset = 0; // Assuming you start writing at the beginning

      // Write data to the file using MPI_File_write_at
      MPI_File_write_at(file, fileoffset, outpsiy[0], (Ny / outstpy) * 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
   }

   // Optional: Synchronize all processes
   MPI_Barrier(MPI_COMM_WORLD);

   return;
}

void outpsi2xz(double complex ***psi, double ***outxz, MPI_File file) {
   long cnti, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Nz / outstpz);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         outxz[cnti / outstpx][cntk / outstpz][0] = x[offsetNx + cnti];
         outxz[cnti / outstpx][cntk / outstpz][1] = z[cntk];
         outxz[cnti / outstpx][cntk / outstpz][2] = cabs(psi[cnti][Ny2][cntk]) * cabs(psi[cnti][Ny2][cntk]);;
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxz, (localNx / outstpx) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outpsi2yz(double complex ***psi, double ***outyz, MPI_File file) {
   long cntj, cntk;
   int rankNx2, offsetNx2;
   MPI_Offset fileoffset;

   rankNx2 = Nx2 / localNx;
   offsetNx2 = Nx2 % localNx;

   fileoffset = 0;

   if (rank == rankNx2) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         for (cntk = 0; cntk < Nz; cntk += outstpz) {
            outyz[cntj / outstpy][cntk / outstpz][0] = y[cntj];
            outyz[cntj / outstpy][cntk / outstpz][1] = z[cntk];
            outyz[cntj / outstpy][cntk / outstpz][2] = cabs(psi[offsetNx2][cntj][cntk]) * cabs(psi[offsetNx2][cntj][cntk]);;
         }
      }

      MPI_File_write_at(file, fileoffset, **outyz, (Ny / outstpy) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
   }
}

void outdenxyz(double complex ***psi, double ***outxyz, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   // MPI IO returns error if the array is too large. As a workaround, we write just Ny * Nz at a time.

   fileoffset = rank * sizeof(double) * localNx * Ny * Nz;

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         for (cntk = 0; cntk < Nz; cntk += outstpz) {
            outxyz[0][cntj][cntk] = cabs(psi[cnti][cntj][cntk]) * cabs(psi[cnti][cntj][cntk]);
         }
      }

      MPI_File_write_at_all(file, fileoffset, **outxyz, (Ny / outstpy) * (Nz / outstpz), MPI_DOUBLE, MPI_STATUS_IGNORE);
      fileoffset += (Ny / outstpy) * (Nz / outstpz) * sizeof(double);
   }

   return;
}

void outpsi(double complex ***psi, MPI_File file) {
   long cnti;
   MPI_Offset fileoffset;

   // MPI IO returns error if the array is too large. As a workaround, we write just Ny * Nz at a time.

   fileoffset = rank * sizeof(double complex) * localNx * Ny * Nz;

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      MPI_File_write_at_all(file, fileoffset, *(psi[cnti]), 2 * (Ny / outstpy) * (Nz / outstpz), MPI_DOUBLE, MPI_STATUS_IGNORE);
      fileoffset += (Ny / outstpy) * (Nz / outstpz) * sizeof(double complex);
   }

   return;
}

void outargxy(double complex ***psi, double ***outxy, MPI_File file) {
   long cnti, cntj;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Ny / outstpy);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         outxy[cnti / outstpx][cntj / outstpy][0] = x[offsetNx + cnti];
         outxy[cnti / outstpx][cntj / outstpy][1] = y[cntj];
         outxy[cnti / outstpx][cntj / outstpy][2] = carg(psi[cnti][cntj][Nz2]);
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxy, (localNx / outstpx) * (Ny / outstpy) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outargxz(double complex ***psi, double ***outxz, MPI_File file) {
   long cnti, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Nz / outstpz);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         outxz[cnti / outstpx][cntk / outstpz][0] = x[offsetNx + cnti];
         outxz[cnti / outstpx][cntk / outstpz][1] = z[cntk];
         outxz[cnti / outstpx][cntk / outstpz][2] = carg(psi[cnti][Ny2][cntk]);
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxz, (localNx / outstpx) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outargyz(double complex ***psi, double ***outyz, MPI_File file) {
   long cntj, cntk;
   int rankNx2, offsetNx2;
   MPI_Offset fileoffset;

   rankNx2 = Nx2 / localNx;
   offsetNx2 = Nx2 % localNx;

   fileoffset = 0;

   if (rank == rankNx2) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         for (cntk = 0; cntk < Nz; cntk += outstpz) {
            outyz[cntj / outstpy][cntk / outstpz][0] = y[cntj];
            outyz[cntj / outstpy][cntk / outstpz][1] = z[cntk];
            outyz[cntj / outstpy][cntk / outstpz][2] = carg(psi[offsetNx2][cntj][cntk]);
         }
      }

      MPI_File_write_at(file, fileoffset, **outyz, (Ny / outstpy) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
   }
}
