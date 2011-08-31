#include <math.h>

#define RHOPREC double

void rho_k(RHOPREC x_vec[][3], int N_x, 
	   RHOPREC k_vec[][3], int N_k,
	   RHOPREC rho_k[][2]){

  int x_i, k_i;
  RHOPREC alpha;

  for(k_i=0; k_i<N_k; k_i++){
    rho_k[k_i][0] = 0.0;
    rho_k[k_i][1] = 0.0;
    for(x_i=0; x_i<N_x; x_i++){
      alpha = \
	x_vec[x_i][0] * k_vec[k_i][0] + 
	x_vec[x_i][1] * k_vec[k_i][1] + 
	x_vec[x_i][2] * k_vec[k_i][2];
      rho_k[k_i][0] += cos(alpha);
      rho_k[k_i][1] += sin(alpha);
    }
  }   
}
