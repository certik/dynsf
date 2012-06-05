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
  register RHOPREC alpha, factor;

  factor = (1.0/sqrt((RHOPREC)N_x));

#pragma omp parallel for	      \
  shared(rho_k, x_vec, k_vec, factor) \
  private(k_i, x_i, alpha)
  for(k_i=0; k_i<N_k; k_i++){
    rho_k[k_i][0] = 0.0;
    rho_k[k_i][1] = 0.0;
    for(x_i=0; x_i<N_x; x_i++){
      alpha =					\
	x_vec[x_i][0] * k_vec[k_i][0] +
	x_vec[x_i][1] * k_vec[k_i][1] +
	x_vec[x_i][2] * k_vec[k_i][2];
      rho_k[k_i][0] += cos(alpha);
      rho_k[k_i][1] += sin(alpha);
    }
    rho_k[k_i][0] *= factor;
    rho_k[k_i][1] *= factor;
  }
}


void rho_j_k(const RHOPREC x_vec[][3], const RHOPREC v_vec[][3], int N_x,
	     const RHOPREC k_vec[][3], int N_k,
	     RHOPREC (* restrict rho_k)[2],
	     RHOPREC (* restrict j_k)[6]){

  int x_i, k_i;
  register RHOPREC alpha, factor, ca, sa;

  factor = (1.0/sqrt((RHOPREC)N_x));

    /* Both rho_k and j_k */
#pragma omp parallel for				\
  shared(rho_k, j_k, x_vec, v_vec, k_vec, factor)	\
  private(k_i, x_i, alpha, ca, sa)
  for(k_i=0; k_i<N_k; k_i++){
    rho_k[k_i][0] = rho_k[k_i][1] = 0.0;
    j_k[k_i][0] = j_k[k_i][1] = j_k[k_i][2] = 0.0;
    j_k[k_i][3] = j_k[k_i][4] = j_k[k_i][5] = 0.0;
    for(x_i=0; x_i<N_x; x_i++){
      alpha =					\
	x_vec[x_i][0] * k_vec[k_i][0] +
	x_vec[x_i][1] * k_vec[k_i][1] +
	x_vec[x_i][2] * k_vec[k_i][2];
      ca = cos(alpha);
      sa = sin(alpha);
      rho_k[k_i][0] += ca;
      rho_k[k_i][1] += sa;
      j_k[k_i][0] += ca * v_vec[x_i][0];
      j_k[k_i][1] += sa * v_vec[x_i][0];
      j_k[k_i][2] += ca * v_vec[x_i][1];
      j_k[k_i][3] += sa * v_vec[x_i][1];
      j_k[k_i][4] += ca * v_vec[x_i][2];
      j_k[k_i][5] += sa * v_vec[x_i][2];
    }
    rho_k[k_i][0] *= factor;
    rho_k[k_i][1] *= factor;
    j_k[k_i][0] *= factor;
    j_k[k_i][1] *= factor;
    j_k[k_i][2] *= factor;
    j_k[k_i][3] *= factor;
    j_k[k_i][4] *= factor;
    j_k[k_i][5] *= factor;
  }
}
