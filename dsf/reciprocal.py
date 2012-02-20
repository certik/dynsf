
# Copyright (C) 2011 Mattias Slabanja <slabanja@chalmers.se>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

__all__ = ['reciprocal']


import sys
from os.path import dirname, join
import numpy as np
from numpy import linalg, array, arange, require, nonzero, pi, sqrt, prod
from ctypes import cdll, byref, c_int, c_float, POINTER


np_f = {'d' : np.float64, 's' : np.float32}
np_c = {'d' : np.complex128, 's' : np.complex64}
np_ndp = np.ctypeslib.ndpointer

_lib = {}
_rho_k = {}
_rho_j_k = {}

ndp_f_2d_r = {}
ndp_c_1d_rw = {}
ndp_c_2d_rw = {}
for t in "ds":
    ndp_f_2d_r[t] = np_ndp(dtype=np_f[t], ndim=2, flags='f_contiguous, aligned')
    ndp_c_1d_rw[t] = np_ndp(dtype=np_c[t], ndim=1,
                            flags='f_contiguous, aligned, writeable')
    ndp_c_2d_rw[t] = np_ndp(dtype=np_c[t], ndim=2,
                            flags='f_contiguous, aligned, writeable')

    _lib[t] = cdll.LoadLibrary(join(dirname(__file__),'_rho_j_k_%s.so' % t))
    _lib[t].rho_k.argtypes = (ndp_f_2d_r[t], c_int,
                              ndp_f_2d_r[t], c_int,
                              ndp_c_1d_rw[t])
    _lib[t].rho_j_k.argtypes = (ndp_f_2d_r[t], ndp_f_2d_r[t], c_int,
                                ndp_f_2d_r[t], c_int,
                                ndp_c_1d_rw[t], ndp_c_2d_rw[t])

def calc_rho_k(x, k, ftype='d'):
    x = require(x, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
    k = require(k, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
    _, Nx = x.shape
    _, Nk = k.shape
    rho_k = np.zeros((Nk,), dtype=np_c[ftype], order='F')
    _lib[ftype].rho_k(x, Nx, k, Nk, rho_k)
    return rho_k

def calc_rho_j_k(x, v, k, ftype='d'):
    assert x.shape == v.shape
    x = require(x, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
    v = require(v, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
    k = require(k, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
    _, Nx = x.shape
    _, Nk = k.shape
    rho_k = np.zeros((Nk,), dtype=np_c[ftype], order='F')
    j_k = np.zeros((3,Nk), dtype=np_c[ftype], order='F')
    _lib[ftype].rho_j_k(x, v, Nx, k, Nk, rho_k, j_k)
    return rho_k, j_k




class reciprocal:
    def __init__(self, box, N_max=-1, k_max=10.0, debug=False, ftype='d'):
        """Creates a set of reciprocal coordinates, and calculate rho_k/j_k
        for a trajectory frame.


        Optionally limit the set to approximately N_max points by
        randomly removing points from a "fully populated grid".
        The points are removed in such a way that for k > k_prune,
        the points will be radially uniformely distributed
        (the value of k_prune is calculated from k_max, N_max,
        and the shape of the box).

        Variables named k_ (such as input argument k_max) are expected
        to be "physicist reciprocal length" (i.e. _with_ 2*pi factor).
        Variables named q_ are expected to be without the 2*pi factor.

        ftype can be either 'd' or 's' (double or single precission)
        """
        assert(N_max == -1 or N_max > 1000)

        self.k_max = k_max
        self.N_max = N_max

        self.ftype = ftype
        npftype = np_f[ftype]
        self.A = require(box.copy(), npftype)
        # B is the "crystallographic" reciprocal vectors
        self.B = linalg.inv(self.A.transpose())

        self.debug = debug

        q_max = k_max/(2.0*pi)
        q_mins = array([linalg.norm(b) for b in self.B])
        q_vol = prod(q_mins)
        self.q_mins = q_mins
        self.q_vol = q_vol

        if N_max == -1 or N_max > pi*q_max**3/(6*q_vol):
            # Use all k-points, do not throw any away
            self.q_prune = None
        else:
            # Use Cardano's formula to find k_prune
            p = -3.0*q_max**2/4
            q = 3.0*N_max*q_vol/pi - q_max**3/4
            D = (p/3)**3 + (q/2)**2
            #assert D < 0.0
            u = (-q/2+sqrt(D+0j))**(1.0/3)
            v = (-q/2-sqrt(D+0j))**(1.0/3)
            x = -(u+v)/2 - 1j*(u-v)*sqrt(3)/2
            self.q_prune = np.real(x) + q_max/2

        b1, b2, b3 = [(2*pi)*x.reshape((3,1,1,1)) for x in self.B]
        N_k1, N_k2, N_k3 = [np.ceil(q_max/dq) for dq in q_mins]
        k_points = \
            b1 * arange(N_k1, dtype=npftype).reshape((1,N_k1,1,   1)) + \
            b2 * arange(N_k2, dtype=npftype).reshape((1,1,   N_k2,1)) + \
            b3 * arange(N_k3, dtype=npftype).reshape((1,1,   1,   N_k3))
        k_points = k_points.reshape((3, k_points.size/3))
        q_distance = sqrt(np.sum(k_points**2, axis=0))*(1.0/(2*pi))

        I, = nonzero(q_distance <= q_max)
        I = I[q_distance[I].argsort()]
        q_distance = q_distance[I]
        k_points = k_points[:,I]    # All k_points < k_max, sorted by length

        if self.q_prune is not None:
            N = len(q_distance)

            # Keep point with probability min(1, (q_prune/|q|)^2) ->
            # aim for an equal number of points per equally thick "onion peel"
            # to even the statistict per radial unit.
            p = np.ones(N)
            p[1:] = (self.q_prune/q_distance[1:])**2

            I, = nonzero(p > np.random.rand(N))
            q_distance = q_distance[I]
            k_points = k_points[:,I]

        self.k_points = k_points
        self.q_distance = q_distance
        self.k_distance = 2.0*pi*q_distance
        N = len(q_distance)
        self.k_direct = k_points.copy()
        self.k_direct[:,1:] /= self.k_distance[1:].reshape((1,N))

    def process_frame(self, frame):
        """Add k-space density to frame

        Calculate the density in k-space for a dynsf-style trajectory frame.
        """
        if self.debug:
            sys.stdout.write("processing frame %i\r" % frame['step'])
            sys.stdout.flush()

        frame = frame.copy()
        if 'vs' in frame:
            rho_ks, j_ks = zip(*[calc_rho_j_k(x, v, self.k_points,
                                              ftype=self.ftype)
                                 for x,v in zip(frame['xs'],frame['vs'])])
            jz_ks = [np.sum(j*self.k_direct, axis=0) for j in j_ks]
            frame['j_ks'] = j_ks
            frame['jz_ks'] = jz_ks
            frame['jpar_ks'] = [j-(jz*self.k_direct) for j,jz in zip(j_ks, jz_ks)]
            frame['rho_ks'] = rho_ks
        else:
            frame['rho_ks'] = [calc_rho_k(x, self.k_points, ftype=self.ftype)
                               for x in frame['xs']]
        return frame

