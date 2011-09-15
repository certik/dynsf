__all__ = ['calc_rho_q', 'calc_rho_j_q']

from ctypes import cdll, byref, c_int, c_float, POINTER
import numpy as np
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


