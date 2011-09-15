#include <math.h>

#define RHOPREC double

void rho_q(RHOPREC x_vec[][3], int N_x, 
	   RHOPREC k_vec[][3], int N_q,
	   RHOPREC rho_q[][2]){

  int x_i, q_i;
  RHOPREC alpha, factor;

  factor = (1.0/sqrt((RHOPREC)N_x));

#pragma omp parallel for	      \
  shared(rho_q, x_vec, k_vec, factor) \
  private(q_i, x_i, alpha)
  for(q_i=0; q_i<N_q; q_i++){
    rho_q[q_i][0] = 0.0;
    rho_q[q_i][1] = 0.0;
    for(x_i=0; x_i<N_x; x_i++){
      alpha =					\
	x_vec[x_i][0] * k_vec[q_i][0] + 
	x_vec[x_i][1] * k_vec[q_i][1] + 
	x_vec[x_i][2] * k_vec[q_i][2];
      rho_q[q_i][0] += cos(alpha);
      rho_q[q_i][1] += sin(alpha);
    }
    rho_q[q_i][0] *= factor;
    rho_q[q_i][1] *= factor;
  }   
}


void rho_j_q(RHOPREC x_vec[][3], RHOPREC v_vec[][3], int N_x, 
	     RHOPREC k_vec[][3], int N_q,
	     RHOPREC rho_q[][2], RHOPREC j_q[][6]){

  int x_i, q_i;
  RHOPREC alpha, factor, ca, sa;

  factor = (1.0/sqrt((RHOPREC)N_x));

    /* Both rho_q and j_q */    
#pragma omp parallel for				\
  shared(rho_q, j_q, x_vec, v_vec, k_vec, factor)	\
  private(q_i, x_i, alpha, ca, sa)
  for(q_i=0; q_i<N_q; q_i++){
    rho_q[q_i][0] = rho_q[q_i][1] = 0.0;
    j_q[q_i][0] = j_q[q_i][1] = j_q[q_i][2] = 0.0;
    j_q[q_i][3] = j_q[q_i][4] = j_q[q_i][5] = 0.0;
    for(x_i=0; x_i<N_x; x_i++){
      alpha =					\
	x_vec[x_i][0] * k_vec[q_i][0] + 
	x_vec[x_i][1] * k_vec[q_i][1] + 
	x_vec[x_i][2] * k_vec[q_i][2];
      ca = cos(alpha);
      sa = sin(alpha);
      rho_q[q_i][0] += ca;
      rho_q[q_i][1] += sa;
      j_q[q_i][0] += ca * v_vec[x_i][0];
      j_q[q_i][1] += sa * v_vec[x_i][0];
      j_q[q_i][2] += ca * v_vec[x_i][1];
      j_q[q_i][3] += sa * v_vec[x_i][1];
      j_q[q_i][4] += ca * v_vec[x_i][2];
      j_q[q_i][5] += sa * v_vec[x_i][2];
    }
    rho_q[q_i][0] *= factor;
    rho_q[q_i][1] *= factor;
    j_q[q_i][0] *= factor;
    j_q[q_i][1] *= factor;
    j_q[q_i][2] *= factor;
    j_q[q_i][3] *= factor;
    j_q[q_i][4] *= factor;
    j_q[q_i][5] *= factor;
  }   
}

