__all__ = ['calc_rho_k', 'calc_rho_j_k']

from ctypes import cdll, byref, c_int, c_float, POINTER
import numpy as np
np_ndp = np.ctypeslib.ndpointer

_lib = cdll.LoadLibrary('./_rho_j_k.so')
_rho_k_d = _lib.rho_k
_rho_j_k_d = _lib.rho_j_k
#_rho_k_s = _lib.rho_k_s
#_rho_j_k_s = _lib.rho_j_k_s

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

_rho_j_k_d.argtypes = [ndp_f64_2d, ndp_f64_2d, c_int, 
                       ndp_f64_2d, c_int,
                       ndp_c128_1d, ndp_c128_2d]
_rho_k_d.argtypes =   [ndp_f64_2d, c_int, 
                       ndp_f64_2d, c_int, 
                       ndp_c128_1d]

#_rho_j_k_s.argtypes = [ndp_f32_2d, ndp_f32_2d, c_int, 
#                       ndp_f32_2d, c_int,
#                       ndp_c64_1d, ndp_c64_2d]
#_rho_k_s.argtypes =   [ndp_f32_2d, c_int, 
#                       ndp_f32_2d, c_int, 
#                       ndp_c64_1d]

def calc_rho_k(x, k):
    x = np.require(x, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    k = np.require(k, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    _, Nx = x.shape
    _, Nk = k.shape
    rho_k = np.zeros((Nk,), dtype=np.complex128, order='F')
    _rho_k_d(x, Nx, k, Nk, rho_k)
    return rho_k

def calc_rho_j_k(x, v, k):
    x = np.require(x, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    v = np.require(x, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    k = np.require(k, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    _, Nx = x.shape
    _, Nk = k.shape
    rho_k = np.zeros((Nk,), dtype=np.complex128, order='F')
    j_k = np.zeros((Nk,3), dtype=np.complex128, order='F')
    _rho_j_k_d(x, v, Nx, k, Nk, rho_k, j_k)
    return rho_k, j_k


