
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
    x = np.require(x, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
    k = np.require(k, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
    _, Nx = x.shape
    _, Nk = k.shape
    rho_k = np.zeros((Nk,), dtype=np_c[ftype], order='F')
    _lib[ftype].rho_k(x, Nx, k, Nk, rho_k)
    return rho_k

def calc_rho_j_k(x, v, k, ftype='d'):
    assert x.shape == v.shape
    x = np.require(x, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
    v = np.require(v, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
    k = np.require(k, np_f[ftype], ['F_CONTIGUOUS', 'ALIGNED'])
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
        randomly removing points. The points are removed in such a way
        that for k>k_prune, the points will be radially uniformely
        distributed (the value of k_prune is calculated from k_max, N_max,
        and the shape of the box).

        k_max should be the "physicist reciprocal length" (with 2*pi factor)

        dtype can be either 'd' or 's' (double or single precission)
        """
        assert(N_max == -1 or N_max > 1000)

        self.k_max = k_max
        self.N_max = N_max

        self.ftype = ftype
        npftype = np_f[ftype]
        self.A = np.require(box.copy(), npftype)
        # B is the "crystallographic" reciprocal vectors
        self.B = np.linalg.inv(self.A.transpose())

        self.debug = debug

        q_max = k_max/(2.0*np.pi)
        q_mins = np.array([np.linalg.norm(b) for b in self.B])
        q_vol = np.prod(q_mins)
        self.q_mins = q_mins
        self.q_vol = q_vol

        if N_max == -1 or N_max > np.pi*q_max**3/(6*q_vol):
            # Do not deselect k-points
            self.q_prune = None
        else:
            # Use Cardano's formula to find k_prune
            p = -3.0*q_max**2/4
            q = 3.0*N_max*q_vol/np.pi - q_max**3/4
            D = (p/3)**3 + (q/2)**2
            #assert D < 0.0
            u = (-q/2+np.sqrt(D+0j))**(1.0/3)
            v = (-q/2-np.sqrt(D+0j))**(1.0/3)
            x = -(u+v)/2 - 1j*(u-v)*np.sqrt(3)/2
            self.q_prune = np.real(x) + q_max/2

        b1, b2, b3 = [(2*np.pi)*x.reshape((3,1,1,1)) for x in self.B]
        Nk1, Nk2, Nk3 = [np.ceil(q_max/dq) for dq in q_mins]
        kvals = \
            b1 * np.arange(Nk1, dtype=npftype).reshape((1,Nk1,1,1)) + \
            b2 * np.arange(Nk2, dtype=npftype).reshape((1,1,Nk2,1)) + \
            b3 * np.arange(Nk3, dtype=npftype).reshape((1,1,1,Nk3))
        kvals = kvals.reshape((3, kvals.size/3))
        qdist = np.sqrt(np.sum(kvals**2, axis=0))*(1.0/(2*np.pi))
        I = np.nonzero(qdist<=q_max)[0]
        I = I[qdist[I].argsort()]
        qdist = qdist[I]
        kvals = kvals[:,I]
        if not self.q_prune is None:
            N = len(qdist)
            p = np.ones(N)
            # N(k) = a k^3
            # N'(k) = 3a k^2
            p[1:] = (self.q_prune/qdist[1:])**2
            I = np.nonzero(p > np.random.rand(N))[0]
            qdist = qdist[I]
            kvals = kvals[:,I]
        self.kvals = kvals
        self.qdist = qdist
        N = len(qdist)
        self.kdist = 2.0*np.pi*qdist
        self.kdirect = kvals / (self.kdist.reshape((1,N)))

    def process_frame(self, frame):
        """Add k-space density to frame

        Calculate the density in k-space for dynsf-style trajectory frame.
        """
        if self.debug:
            sys.stdout.write("processing frame %i\r" % frame['step'])
            sys.stdout.flush()

        frame = frame.copy()
        if 'vs' in frame:
            rho_ks, j_ks = zip(*[calc_rho_j_k(x, v, self.kvals,
                                              ftype=self.ftype)
                                 for x,v in zip(frame['xs'],frame['vs'])])
            jz_ks = [np.sum(j*self.kdirect, axis=0) for j in j_ks]
            frame['j_ks'] = j_ks
            frame['jz_ks'] = jz_ks
            frame['jpar_ks'] = [j-(jz*self.kdirect) for j,jz in zip(j_ks, jz_ks)]
            frame['rho_ks'] = rho_ks
        else:
            frame['rho_ks'] = [calc_rho_k(x, self.kvals,
                                          ftype=self.ftype)
                               for x in frame['xs']]
        return frame

