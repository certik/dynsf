#include <math.h>

// i3: L1 2x32 kB
//     L2 2x256 kB
//
// 64x64 double = 32kB
#define NBLK 60

#define RHOPREC double

void rho_k(RHOPREC x_vec[][3], int N_x, 
	   RHOPREC k_vec[][3], int N_k,
	   RHOPREC rho_k[][2]){

  int x_i, k_i, x_ii, k_ii;
  RHOPREC alpha, factor;

  factor = (1.0/sqrt((RHOPREC)N_x));

#define INNER(xi,ki) \
  alpha = \
    x_vec[xi][0] * k_vec[ki][0] + \
    x_vec[xi][1] * k_vec[ki][1] + \
    x_vec[xi][2] * k_vec[ki][2];  \
  rho_k[ki][0] += cos(alpha); \
  rho_k[ki][1] += sin(alpha);


#pragma omp parallel for \
  shared(rho_k, x_vec, k_vec, factor) \
  private(k_i, x_i, k_ii, x_ii, alpha)
  for(k_i=0; k_i<N_k-NBLK; k_i+=NBLK){
    for(x_i=0; x_i<N_x-NBLK; x_i+NBLK){
      for(k_ii=k_i; k_ii<k_i+NBLK; k_ii++){
	rho_k[k_ii][0] = 0.0;
	rho_k[k_ii][1] = 0.0;
	for(x_ii=x_i; x_ii<x_i+NBLK; x_ii++) { INNER(x_ii, k_ii); }
      }
    }
    for(k_ii=k_i; k_ii<k_i+NBLK; k_ii++){
      for(x_ii=x_i+NBLK; x_ii<N_x; x_ii++){ INNER(x_ii, k_ii); }
      rho_k[k_i][0] *= factor;
      rho_k[k_i][1] *= factor;
    }
  }
  for(x_i=0; x_i<N_x-NBLK; x_i+NBLK){
    for(k_ii=k_i+NBLK; k_ii<N_k; k_ii++){
      rho_k[k_ii][0] = 0.0;
      rho_k[k_ii][1] = 0.0;
      for(x_ii=x_i; x_ii<x_i+NBLK; x_ii++) { INNER(x_ii, k_ii); }
    }
  }
  for(k_ii=k_i+NBLK; k_ii<N_k; k_ii++){
    for(x_ii=x_i+NBLK; x_ii<N_x; x_ii++){ INNER(x_ii, k_ii); }
    rho_k[k_i][0] *= factor;
    rho_k[k_i][1] *= factor;
  }
}

