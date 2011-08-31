#!/usr/bin/env python

from ctypes import cdll, byref, c_int, c_float, POINTER
from ctypes.util import find_library
from itertools import takewhile

import numpy as np


lname = find_library('gmx')
libgmx = lname and cdll.LoadLibrary(lname)
libgmx.open_xtc.restype = POINTER(c_int)

#
#class Frame:
#    def __init__(self, x, box, step, time):
#        pass
#    

class XTC_iter:
    """Iterate through an xtc-file

    Allows iteration through an xtc file, frame by frame.
    Each iteration yields a dictionary 
    {'N': number of atoms,
     'box': simulation box as 3 row vectors (nm),
     'x': xyz data as 3xN array (nm),
     'step': simulation step,
     'time': simulation time (ps) 
    """
    def __init__(self, fn, max_frames=-1):
        if libgmx is None:
            raise RuntimeError("No libgmx found!")

        self._fio = libgmx.open_xtc(fn, 'r')
        if not self._fio:
            raise IOError("blah, failed to open %s" % fn)

        # single prec gmx-real equals float, right?
        self._natoms = c_int()
        self._step =   c_int()
        self._time =   c_float()
        self._box =    (c_float*9)() # typedef real matrix[DIM][DIM];
        self._x =      POINTER(c_float)() # typedef real rvec[DIM];
        self._prec =   c_float()
        self._bOK =    c_int()  # gmx_bool equals int
        self._first_done = False
        self._open = True
        self._frame_counter = 0
        self.max_frames = max_frames

    def _get_first(self):
        # Read first frame, update state of object
        res = libgmx.read_first_xtc(self._fio, 
                                    byref(self._natoms), 
                                    byref(self._step), 
                                    byref(self._time), 
                                    byref(self._box), 
                                    byref(self._x), 
                                    byref(self._prec), 
                                    byref(self._bOK))
        self._first_done = True
        if not res:
            raise IOError("read_first_xtc failed")
        if not self._bOK.value:
            raise IOError("corrupt frame in xtc-file?")
        return True
    
    def _get_next(self):
        # Try to read the next frame, update state of object
        res = libgmx.read_next_xtc(self._fio, 
                                   self._natoms.value, 
                                   byref(self._step), 
                                   byref(self._time), 
                                   byref(self._box), 
                                   self._x,
                                   byref(self._prec), 
                                   byref(self._bOK))
        if not res:
            return False
        if not self._bOK.value:
            raise IOError("corrupt frame in xtc-file?")
        return True
    
    def __iter__(self):
        return self
    
    def close(self):
        if self._open:
            libgmx.close_xtc(self._fio)
            self._open = False
        
    def next(self):
        if self.max_frames == self._frame_counter or not self._open:
            raise StopIteration
        self._frame_counter += 1

        if not self._first_done:
            self._get_first()
        else:
            if not self._get_next():
                self.close()
                raise StopIteration
            
        N = self._natoms.value
        return {'N' : N,
                'box' : np.array(self._box[0:9]).reshape((3,3), order='F'),
                'x' : np.array(self._x[0:3*N]).reshape((3,N), order='F'), 
                'step' : self._step.value,
                'time' : self._time.value}



class cyclic_list(list):
    def __getitem__(self,key):
        return super(cyclic_list, self).__getitem__(key%len(self))
    def __setitem__(self,key,val):
        return super(cyclic_list, self).__setitem__(key%len(self),val)
    def __getslice__(self,i,j):
        n = len(self)
        return [self[x%n] for x in range(i,j)]


librho_k = cdll.LoadLibrary('./librho_k.so')
_rho_k = librho_k.rho_k
_rho_k.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, 
                                          flags='f_contiguous, aligned'),
                   c_int, 
                   np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, 
                                          flags='f_contiguous, aligned'),
                   c_int,
                   np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, 
                                          flags='f_contiguous, aligned, writeable')]
def calc_rho_k(x, k):
    x = np.require(x, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    k = np.require(k, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    _, Nx = x.shape
    _, Nk = k.shape
    rho_k = np.zeros((Nk,), dtype=np.complex128, order='F')
    _rho_k(x, Nx, k, Nk, rho_k)
    return (1.0/np.sqrt(Nx))*rho_k


def weighted_average(x,y):
    pass


class averager:
    def __init__(self, init_array, N_slots):
        assert type(N_slots) is int
        self._N = N_slots
        self._data = [np.array(init_array) for n in range(N_slots)]
        self._samples = [0]*N_slots
    def add(self, array, slot):
        self._data[slot] = self._data[slot] + array
        self._samples[slot] += 1
    def get_av(self, slot):
        n = self._samples[slot]
        assert n > 0
        f = 1.0/n
        return f*self._data[slot]

if __name__ == '__main__':
    import sys
    import optparse
    from math import ceil

    parser = optparse.OptionParser()
    parser.add_option('-f', '', metavar='XTC_FILE',
                      help='Trajectory file in xtc format')
    parser.add_option('-n', '', metavar='INDEX_FILE',
                      help='Index file (Gromacs style) for specifying '
                      'atom types. If none is given, all atoms will be '
                      'considered identical.')
    parser.add_option('','--tc', metavar='CORR_TIME', type='float',
                      help='Correlation time (ps) to consider.')
    parser.add_option('','--nc', metavar='CORR_STEPS', type='int',
                      help='Number of time correlation steps to consider.')
    parser.add_option('','--nk', metavar='KPOINTS', type='int',
                      default=20,
                      help='Number of discrete spatial points')
    parser.add_option('','--max-frames', metavar='NFRAMES', type='int',
                      default=-1,
                      help='Read no more than NFRAMES frames from trajectory file')
    parser.add_option('-v', '--verbose', action='store_true', 
                      default=False,
                      help='Verbose output')

    (options, args) = parser.parse_args()
    if not options.f:
        parser.print_help()
        sys.exit()

    if options.tc and options.nc:
        print('Options --tc and --nc are mutuly exclusive')
        sys.exit(1)

    traj = XTC_iter(options.f)
    f0 = traj.next()
    reference_box = f0['box']
    f1 = traj.next()
    delta_t = f1['time'] - f0['time']
    if options.verbose: 
        print('delta_t found to be %f [ps] --> f_max = %f [GHz]' % \
                  (delta_t, 1000.0/delta_t))
    traj.close()

    
    if options.tc:
        assert(options.tc >= 0.0)
        N_tc = 1 + int(ceil(options.tc/delta_t))
        N_tc += N_tc%2
    elif options.nc:
        assert(options.nc >= 1)
        N_tc = options.nc
    else:
        N_tc = 1

    if options.verbose:
        if N_tc > 1:
            tc = delta_t*(N_tc-1)
            print('tc is %f [ps] --> f_min = %f [GHz]' % \
                      (tc, 1000.0/tc))
            print('0..f_max covered by %i points' % N_tc)

    if options.verbose:
        print('Simulation box = \n%s' % str(reference_box))

    a1, a2, a3 = reference_box
    nr = np.linalg.norm
    cr = np.cross
    vol = abs(np.dot(cr(a1,a2),a3))
    assert abs(vol - nr(a1)*nr(a2)*nr(a3)) <= np.finfo(vol).eps
    b1 = (2*np.pi)*cr(a2,a3)/vol
    b2 = (2*np.pi)*cr(a3,a1)/vol
    b3 = (2*np.pi)*cr(a1,a2)/vol
    # Temporary ugly...
    b2 = b2 * (np.linalg.norm(b1)/np.linalg.norm(b2))
    b3 = b3 * (np.linalg.norm(b1)/np.linalg.norm(b3))
    delta_k = np.linalg.norm(b1)
    Nk = options.nk
    max_k = delta_k*Nk
    if options.verbose:
        print('N = %i --> delta_x = %f [nm]' % (Nk, nr(a1)/Nk))

    b1 = b1.reshape((3,1,1,1))
    b2 = b2.reshape((3,1,1,1))
    b3 = b3.reshape((3,1,1,1))

    kvals = \
        b1 * np.arange(Nk, dtype=np.float64).reshape((1,Nk,1,1)) + \
        b2 * np.arange(Nk, dtype=np.float64).reshape((1,1,Nk,1)) + \
        b3 * np.arange(Nk, dtype=np.float64).reshape((1,1,1,Nk))
    kvals = kvals.reshape((3, kvals.size/3)).transpose()
    kdist = np.sqrt(np.sum(kvals**2, axis=1))
    I = np.nonzero(kdist<=max_k)[0]
    I = I[kdist[I].argsort()]
    kdist = kdist[I]
    kvals = kvals[I,].transpose()

    def add_rho_k(frame):
        frame['rho_k'] = calc_rho_k(frame['x'], kvals)
        return frame

    traj = XTC_iter(options.f, max_frames=options.max_frames)
    try:
        frame_list = cyclic_list([add_rho_k(traj.next()) for x in range(N_tc)])
    except StopIteration:
        print('Failed to read %i frames (minimum required) from %s' % \
                  (N_tc, options.f))
        sys.exit(1)

    # * Assert box is not changed during consecutive frames
    # * Handle different time steps?
    # * Handle none cubic box in a good way
    # * Assert box is square

    
    F_k_t_av = averager(np.zeros(len(kdist)), N_tc)

    for f_index, frame in enumerate(traj):
        frame_list[f_index+N_tc-1] = add_rho_k(frame)
        if options.verbose: print(frame['step'])

        rho_k_0 = frame_list[f_index]['rho_k']
        for i in range(N_tc):
            rho_k_i = frame_list[f_index+i]['rho_k']
            F_k_t_av.add(np.real(rho_k_0*rho_k_i.conjugate()), i)
                
    F_k_t_full = np.array([F_k_t_av.get_av(i) for i in range(N_tc)])

    # dirty smooth it out
    pts = 2*Nk
    rng = (-delta_k/4, max_k+delta_k/4) 
    Npoints, edges = np.histogram(kdist, bins=pts, range=rng)
    F_k_t = np.zeros((N_tc, pts))
    F_k_t_sd = np.zeros((N_tc, pts))
    k = 0.5*(edges[1:]+edges[:-1])
    t = delta_t*np.arange(N_tc)
    
    ci = 0
    for i, n in enumerate(Npoints):
        if n == 0:
            F_k_t[:,i] = np.NaN
            continue
        s = F_k_t_full[:,ci:ci+n]
        F_k_t[:,i] = np.mean(s, axis=1)
        F_k_t_sd[:,i] = np.std(s, axis=1)
        ci += n
