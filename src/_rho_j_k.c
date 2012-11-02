/*
 Copyright (C) 2011 Mattias Slabanja <slabanja@chalmers.se>

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 02110-1301, USA.
*/

#include <math.h>

#ifndef RHOPREC
#warning "Defaulting to double precission"
#define RHOPREC double
#endif

void rho_k(const RHOPREC x_vec[][3], int N_x,
           const RHOPREC k_vec[][3], int N_k,
           RHOPREC (* restrict rho_k)[2]){

  int x_i, k_i;
  RHOPREC rho_ki_0, rho_ki_1;
  RHOPREC factor = (1.0 / sqrt((RHOPREC)N_x));
  register RHOPREC alpha;

#ifdef _OPENACC
#pragma acc kernels copyin(x_vec[0:N_x][0:3],k_vec[0:N_k][0:3]) copyout(rho_k[0:N_k][0:2])
  {
#endif

#pragma omp parallel for \
  shared(rho_k, x_vec, k_vec, factor) \
  private(k_i, x_i, alpha, rho_ki_0, rho_ki_1)

  for(k_i=0; k_i<N_k; k_i++){

    rho_ki_0 = 0.0;
    rho_ki_1 = 0.0;

    for(x_i=0; x_i<N_x; x_i++){
      alpha = \
        x_vec[x_i][0] * k_vec[k_i][0] +
        x_vec[x_i][1] * k_vec[k_i][1] +
        x_vec[x_i][2] * k_vec[k_i][2];
      rho_ki_0 += cos(alpha);
      rho_ki_1 += sin(alpha);
    }
    rho_k[k_i][0] = factor * rho_ki_0;
    rho_k[k_i][1] = factor * rho_ki_1;
  }

#ifdef _OPENACC
  }
#endif
}


void rho_j_k(const RHOPREC x_vec[][3], const RHOPREC v_vec[][3], int N_x,
             const RHOPREC k_vec[][3], int N_k,
             RHOPREC (* restrict rho_k)[2],
             RHOPREC (* restrict j_k)[6]){

  int x_i, k_i;
  RHOPREC rho_ki_0, rho_ki_1;
  RHOPREC j_ki_0, j_ki_1, j_ki_2, j_ki_3, j_ki_4, j_ki_5;
  RHOPREC factor = (1.0 / sqrt((RHOPREC)N_x));
  register RHOPREC alpha, ca, sa;

#ifdef _OPENACC
#pragma acc kernels copyin(x_vec[0:N_x][0:3], v_vec[0:N_x][0:3], k_vec[0:N_k][0:3]) \
                    copyout(rho_k[0:N_k][0:2], j_k[0:N_k][0:6])
  {
#endif

    /* Both rho_k and j_k */
#pragma omp parallel for \
  shared(rho_k, j_k, x_vec, v_vec, k_vec, factor) \
  private(k_i, x_i, alpha, ca, sa, rho_ki_0, rho_ki_1, \
          j_ki_0, j_ki_1, j_ki_2, j_ki_3, j_ki_4, j_ki_5)
  for(k_i=0; k_i<N_k; k_i++){

    rho_ki_0 = 0.0;
    rho_ki_1 = 0.0;
    j_ki_0 = 0.0;
    j_ki_1 = 0.0;
    j_ki_2 = 0.0;
    j_ki_3 = 0.0;
    j_ki_4 = 0.0;
    j_ki_5 = 0.0;

    for(x_i=0; x_i<N_x; x_i++){
      alpha = \
        x_vec[x_i][0] * k_vec[k_i][0] +
        x_vec[x_i][1] * k_vec[k_i][1] +
        x_vec[x_i][2] * k_vec[k_i][2];
      ca = cos(alpha);
      sa = sin(alpha);
      rho_ki_0 += ca;
      rho_ki_1 += sa;
      j_ki_0 += ca * v_vec[x_i][0];
      j_ki_1 += sa * v_vec[x_i][0];
      j_ki_2 += ca * v_vec[x_i][1];
      j_ki_3 += sa * v_vec[x_i][1];
      j_ki_4 += ca * v_vec[x_i][2];
      j_ki_5 += sa * v_vec[x_i][2];
    }
    rho_k[k_i][0] = factor * rho_ki_0;
    rho_k[k_i][1] = factor * rho_ki_1;
    j_k[k_i][0] = factor * j_ki_0;
    j_k[k_i][1] = factor * j_ki_1;
    j_k[k_i][2] = factor * j_ki_2;
    j_k[k_i][3] = factor * j_ki_3;
    j_k[k_i][4] = factor * j_ki_4;
    j_k[k_i][5] = factor * j_ki_5;
  }

#ifdef _OPENACC
  }
#endif
}
