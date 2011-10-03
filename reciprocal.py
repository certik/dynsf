__all__ = ['reciprocal']


import sys
import numpy as np
from ctypes import cdll, byref, c_int, c_float, POINTER

np_ndp = np.ctypeslib.ndpointer

_lib = cdll.LoadLibrary('./_rho_j_q.so')
_rho_q_d = _lib.rho_q
_rho_j_q_d = _lib.rho_j_q
#_rho_q_s = _lib.rho_q_s
#_rho_j_q_s = _lib.rho_j_q_s

ndp_f64_2d = np_ndp(dtype=np.float64, ndim=2, 
                    flags='f_contiguous, aligned')
ndp_c128_1d = np_ndp(dtype=np.complex128, ndim=1, 
                     flags='f_contiguous, aligned, writeable')
ndp_c128_2d = np_ndp(dtype=np.complex128, ndim=2, 
                     flags='f_contiguous, aligned, writeable')

ndp_f32_2d = np_ndp(dtype=np.float32, ndim=2, 
                    flags='f_contiguous, aligned')
ndp_c64_1d = np_ndp(dtype=np.complex64, ndim=1, 
                    flags='f_contiguous, aligned, writeable')
ndp_c64_2d = np_ndp(dtype=np.complex64, ndim=2, 
                    flags='f_contiguous, aligned, writeable')

_rho_j_q_d.argtypes = [ndp_f64_2d, ndp_f64_2d, c_int, 
                       ndp_f64_2d, c_int,
                       ndp_c128_1d, ndp_c128_2d]
_rho_q_d.argtypes =   [ndp_f64_2d, c_int, 
                       ndp_f64_2d, c_int, 
                       ndp_c128_1d]

#_rho_j_q_s.argtypes = [ndp_f32_2d, ndp_f32_2d, c_int, 
#                       ndp_f32_2d, c_int,
#                       ndp_c64_1d, ndp_c64_2d]
#_rho_q_s.argtypes =   [ndp_f32_2d, c_int, 
#                       ndp_f32_2d, c_int, 
#                       ndp_c64_1d]

def calc_rho_q(x, q):
    x = np.require(x, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    q = np.require(q, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    _, Nx = x.shape
    _, Nq = q.shape
    rho_q = np.zeros((Nq,), dtype=np.complex128, order='F')
    _rho_q_d(x, Nx, q, Nq, rho_q)
    return rho_q

def calc_rho_j_q(x, v, q):
    assert x.shape == v.shape
    x = np.require(x, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    v = np.require(v, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    q = np.require(q, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    _, Nx = x.shape
    _, Nq = q.shape
    rho_q = np.zeros((Nq,), dtype=np.complex128, order='F')
    j_q = np.zeros((3,Nq), dtype=np.complex128, order='F')
    _rho_j_q_d(x, v, Nx, q, Nq, rho_q, j_q)
    return rho_q, j_q




class reciprocal:
    def __init__(self, box, N_max=-1, q_max=10.0, debug=False):
        """Creates a set of reciprocal coordinates, and calculate rho_q/j_q 
        for a trajectory frame.

        Optionally limit the set to approximately N_max points by
        randomly removing points. The points are removed in such a way
        that for q>q_prune, the points will be radially uniformely 
        distributed (the value of q_prune is calculated from q_max, N_max,
        and the shape of the box). 

        q_max should be the "crystallographic reciprocal length" (no 2*pi factor)
        """
        assert(N_max == -1 or N_max > 1000)

        self.q_max = q_max
        self.N_max = N_max
        self.A = box.copy()
        # B is the "crystallographic" reciprocal vectors
        self.B = np.linalg.inv(self.A.transpose())

        self.debug = debug

        q_mins = np.array([np.linalg.norm(b) for b in self.B])
        q_vol = np.prod(q_mins)
        self.q_mins = q_mins
        self.q_vol = q_vol

        if N_max == -1 or N_max > np.pi*q_max**3/(6*q_vol):
            # Do not deselect k-points
            self.q_prune = None
        else:
            # Use Cardano's formula to find q_prune
            p = -3.0*q_max**2/4
            q = 3.0*N_max*q_vol/np.pi - q_max**3/4
            D = (p/3)**3 + (q/2)**2
            #assert D < 0.0
            u = (-q/2+np.sqrt(D+0j))**(1.0/3)
            v = (-q/2-np.sqrt(D+0j))**(1.0/3)
            x = -(u+v)/2 - 1j*(u-v)*np.sqrt(3)/2
            self.q_prune = np.real(x) + q_max/2

        b1, b2, b3 = [(2*np.pi)*x.reshape((3,1,1,1)) for x in self.B]
        Nk1, Nk2, Nk3 = [np.ceil(q_max/dk) for dk in q_mins]
        kvals = \
            b1 * np.arange(Nk1, dtype=np.float64).reshape((1,Nk1,1,1)) + \
            b2 * np.arange(Nk2, dtype=np.float64).reshape((1,1,Nk2,1)) + \
            b3 * np.arange(Nk3, dtype=np.float64).reshape((1,1,1,Nk3))
        kvals = kvals.reshape((3, kvals.size/3))
        qdist = np.sqrt(np.sum(kvals**2, axis=0))*(1.0/(2*np.pi))
        I = np.nonzero(qdist<=q_max)[0]
        I = I[qdist[I].argsort()]
        qdist = qdist[I[1:]]
        kvals = kvals[:,I[1:]]
        if not self.q_prune is None:
            N = len(qdist)
            p = np.ones(N)
            # N(k) = a k^3
            # N'(k) = 3a k^2
            p = (self.q_prune/qdist)**2
            I = np.nonzero(p > np.random.rand(N))[0]
            qdist = qdist[I]
            kvals = kvals[:,I]
        self.kvals = kvals
        self.qdist = qdist
        N = len(qdist)
        self.kdirect = kvals / (2.0*np.pi*qdist.reshape((1,N)))

    def process_frame(self, frame):
        if self.debug: 
            sys.stdout.write("processing frame at time = %f\r" % frame['time'])
            sys.stdout.flush()

        frame = frame.copy()
        if 'vs' in frame:
            rho_qs, j_qs = zip(*[calc_rho_j_q(x, v, self.kvals) 
                                 for x,v in zip(frame['xs'],frame['vs'])])
            jz_qs = [np.sum(j*self.kdirect, axis=0) for j in j_qs]
            frame['j_qs'] = j_qs
            frame['jz_qs'] = jz_qs
            frame['jpar_qs'] = [j-(jz*self.kdirect) for j,jz in zip(j_qs, jz_qs)]
            frame['rho_qs'] = rho_qs
        else:
            frame['rho_qs'] = [calc_rho_q(x, self.kvals) for x in frame['xs']]
        return frame

