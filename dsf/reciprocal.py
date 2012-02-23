
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

__all__ = ['reciprocal_isotropic', 'reciprocal_line']


import sys
from os.path import dirname, join
import numpy as np
import logging
from numpy import linalg, array, arange, require, nonzero, pi, sqrt, prod
from ctypes import cdll, byref, c_int, c_float, POINTER

logger = logging.getLogger('dynsf')

np_f = dict(d=np.float64, s=np.float32)
np_c = dict(d=np.complex128, s=np.complex64)
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


def get_prune_distance(max_points, max_q, vol_q):
    """Return the prune distance for q/k-points in the isotropic case

    max_points corresponds to the wanted number of resulting q/k-points,
    max_q corresponds to the maximum q-value in the resulting q/k-point set,
    vol_q corresponds to the q-space volume for a single q-point.
    If points are selected from the full grid with probability
    min(1, (q_prune/|q|)^2), k-space will on average be sampled with
    an equal number of points per radial unit (for q > q_prune).
    """
    # Use Cardano's formula to find k_prune
    p = -3.0*max_q**2/4
    q = 3.0*max_points*q_vol/pi - max_q**3/4
    D = (p/3)**3 + (q/2)**2

    u = (-q/2+sqrt(D+0j))**(1.0/3)
    v = (-q/2-sqrt(D+0j))**(1.0/3)
    x = -(u+v)/2 - 1j*(u-v)*sqrt(3)/2
    return np.real(x) + max_q/2

class reciprocal_line:
    def __init__(self, points=1000, kdirection=(1.0,1.0,1.0), ftype='d'):

        self.ftype = ftype
        npftype = np_f[ftype]
        kdirection = require(kdirection, npftype).reshape((3,1))
        self.k_points = kdirection * np.linspace(0.0, 1.0, points)
        self.k_distance = sqrt(np.sum(self.k_points**2, axis=0))
        self.q_distance = self.k_distance * (1.0/(2*pi))
        self.k_direct = self.k_points.copy()
        self.k_direct[:,1:] /= self.k_distance[1:].reshape((1,points-1))

    def process_frame(self, frame):
        logger.debug("processing frame step %i" % frame['step'])
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



class reciprocal_isotropic:
    def __init__(self, box, max_points=10000, max_k=10.0, ftype='d'):
        """Creates a set of reciprocal coordinates suitable for isotropic
        sampling of k-space. Provide a method to calculate rho_k/j_k
        for trajectory frames.


        Optionally limit the set to approximately max_points points by
        randomly removing points from a "fully populated grid".
        The points are removed in such a way that for k > k_prune,
        the points will be radially uniformely distributed
        (the value of k_prune is calculated from max_k, max_points,
        and the shape of the box).

        Variables named k-something (such as input argument max_k) are expected
        to be "physicist reciprocal length" (i.e. _with_ 2*pi factor).
        Variables named q-something are expected to be without the 2*pi factor.

        ftype can be either 'd' or 's' (double or single precission)
        """

        assert(max_points > 1000)

        self.max_k = max_k
        self.max_points = max_points

        self.ftype = ftype
        npftype = np_f[ftype]
        self.A = require(box.copy(), npftype)
        # B is the "crystallographic" reciprocal vectors
        self.B = linalg.inv(self.A.transpose())

        max_q = max_k/(2.0*pi)
        q_mins = array([linalg.norm(b) for b in self.B])
        q_vol = prod(q_mins)
        self.q_mins = q_mins

        if max_points > pi*max_q**3/(6*q_vol):
            # Use all k-points, do not throw any away
            self.q_prune = None
        else:
            self.q_prune = get_prune_distance(max_points, max_q, q_vol)

        b1, b2, b3 = [(2*pi)*x.reshape((3,1,1,1)) for x in self.B]
        N_k1, N_k2, N_k3 = [np.ceil(max_q/dq) for dq in q_mins]
        k_points = \
            b1 * arange(N_k1, dtype=npftype).reshape((1,N_k1,1,   1)) + \
            b2 * arange(N_k2, dtype=npftype).reshape((1,1,   N_k2,1)) + \
            b3 * arange(N_k3, dtype=npftype).reshape((1,1,   1,   N_k3))
        k_points = k_points.reshape((3, k_points.size/3))
        q_distance = sqrt(np.sum(k_points**2, axis=0))*(1.0/(2*pi))

        I, = nonzero(q_distance <= max_q)
        I = I[q_distance[I].argsort()]
        q_distance = q_distance[I]
        k_points = k_points[:,I]    # All k_points < max_k, sorted by length

        if self.q_prune is not None:
            N = len(q_distance)

            # Keep point with probability min(1, (q_prune/|q|)^2) ->
            # aim for an equal number of points per equally thick "onion peel"
            # to get equal number of points per radial unit.
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
        self.k_direct[:,1:] /= self.k_distance[1:].reshape((1,N-1))


    def process_frame(self, frame):
        """Add k-space density to frame

        Calculate the density in k-space for a dynsf-style trajectory frame.
        """
        logger.debug("processing frame step %i" % frame['step'])
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

