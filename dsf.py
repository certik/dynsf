#!/usr/bin/env python

from ctypes import cdll, byref, c_int, c_float, POINTER
from ctypes.util import find_library
from itertools import takewhile, count

import numpy as np
import re


lname = find_library('gmx')
libgmx = lname and cdll.LoadLibrary(lname)
if libgmx: 
    # single prec gmx-real equals float, right?
    xtcfloat_np = np.float32
    xtcfloat_ct = c_float
    xtcint_ct = c_int

    # t_fileio *open_xtc(const char *filename,const char *mode);
    # /* Open a file for xdr I/O */
    libgmx.open_xtc.restype = POINTER(xtcint_ct)

    # int read_first_xtc(t_fileio *fio,
    #                           int *natoms,int *step,real *time,
    #                           matrix box,rvec **x,real *prec,gmx_bool *bOK);
    # /* Open xtc file, read xtc file first time, allocate memory for x */
    libgmx.read_first_xtc.restype = xtcint_ct
    libgmx.read_first_xtc.argtypes = [
        POINTER(xtcint_ct), POINTER(xtcint_ct), 
        POINTER(xtcint_ct), POINTER(xtcfloat_ct), 
        np.ctypeslib.ndpointer(dtype=xtcfloat_np, shape=(3,3), 
                               flags='f_contiguous, aligned'),
        POINTER(POINTER(xtcfloat_ct)),
        POINTER(xtcfloat_ct), POINTER(xtcint_ct)]
    
    # int read_next_xtc(t_fileio *fio,
    #                          int natoms,int *step,real *time,
    #                          matrix box,rvec *x,real *prec,gmx_bool *bOK);
    # /* Read subsequent frames */
    libgmx.read_next_xtc.restype = xtcint_ct
    libgmx.read_next_xtc.argtypes = [
        POINTER(xtcint_ct), xtcint_ct,
        POINTER(xtcint_ct), POINTER(xtcfloat_ct), 
        np.ctypeslib.ndpointer(dtype=xtcfloat_np, shape=(3,3), 
                               flags='f_contiguous, aligned'),
        np.ctypeslib.ndpointer(dtype=xtcfloat_np, ndim=2, 
                               flags='f_contiguous, aligned'),
        POINTER(xtcfloat_ct), POINTER(xtcint_ct)]
    

class XTC_reader:
    """Iterable object

    Iterate through the frames of an xtc-file.
    Each frame is represented as a dictionary.
    {'N': number of atoms,
     'box': simulation box as 3 row vectors (nm),
     'x': xyz data as 3xN array (nm),
     'step': simulation step,
     'time': simulation time (ps) }
    """
    def __init__(self, file_name, max_frames=-1, index_file=None):
        if libgmx is None:
            raise RuntimeError("No libgmx found, can't read xtc-file")
        
        self._fio = libgmx.open_xtc(file_name, 'r')
        if not self._fio:
            raise IOError("Failed to open file %s (for some reason)" % file_name)

        self._natoms = xtcint_ct()
        self._step =   xtcint_ct()
        self._time =   xtcfloat_ct()
        self._box =    np.require(np.zeros((3,3)),
                                  xtcfloat_np, ['F_CONTIGUOUS', 'ALIGNED'])
        self._x =      None
        self._prec =   xtcfloat_ct()
        self._bOK =    xtcint_ct()  # gmx_bool equals int
        self._open = True
        self._first_called = False
        self._frame_counter = 0
        self.max_frames = max_frames

        self.index_file = index_file

        
    def _setup_indexes(self):
        # This requires the knowledge of the number of atoms, N.
        # Hence, it cannot be called until after the first frame is read.
        N = self._natoms.value
        if self.index_file:
            self.types = []
            self.indexes = []
            for t, I in read_ndx_file(self.index_file):
                if I[0]<0 or I[-1]>=N:
                    raise RuntimeError('Invalid index found in index file')
                self.types.append(t)
                self.indexes.append(I)
        else:
            self.types = ['all']
            self.indexes = [np.arange(N)]

    def _get_first(self):
        # Read first frame, update state of self, create indexes etc
        _xfirst = POINTER(xtcfloat_ct)()
        res = libgmx.read_first_xtc(self._fio, self._natoms, 
                                    self._step, self._time, 
                                    self._box, _xfirst,
                                    self._prec, self._bOK)
        self._first_called = True
        if not res:
            raise IOError("read_first_xtc failed")
        if not self._bOK.value:
            raise IOError("corrupt frame in xtc-file?")

        N = self._natoms.value
        self._x = np.require(np.array(_xfirst[0:3*N]).reshape((3,N), order='F'),
                             xtcfloat_np, ['F_CONTIGUOUS', 'ALIGNED'])
        
        self._setup_indexes()
    
    def _get_next(self):
        # get next frame, update state of self
        res = libgmx.read_next_xtc(self._fio, self._natoms.value, 
                                   self._step, self._time,
                                   self._box, self._x,
                                   self._prec, self._bOK)
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

        if not self._first_called:
            self._get_first()
        else:
            if not self._get_next():
                self.close()
                raise StopIteration
            
        xs = [self._x[:,I] for I in self.indexes]
        return {'N' : self._natoms.value,
                'box' : self._box.copy('F'),
                'step' : self._step.value,
                'time' : self._time.value,
                'xs' : xs
                }



def read_ndx_file(file_name):
    # Read an ini-style gromacs index file
    section_re = re.compile(r'^ *\[ *([a-zA-Z0-9_.-]+) *\] *$')
    sections = []
    members = []
    name = None
    with open(file_name, 'r') as f:
        for L in f:
            m = section_re.match(L)
            if m:
                if members and name:
                    sections.append((name, np.unique(np.array(members))-1))
                name = m.group(1)
                members = []
            elif not L.isspace():
                members += map(int, L.split())
        if members and name:
            sections.append((name, np.unique(np.array(members))-1))
    return sections



class cyclic_list(list):
    def __getitem__(self,key):
        return super(cyclic_list, self).__getitem__(key%len(self))
    def __setitem__(self,key,val):
        return super(cyclic_list, self).__setitem__(key%len(self),val)
    def __getslice__(self,i,j):
        n = len(self)
        return [self[x] for x in range(i,j)]


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
    return rho_k




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

class reciprocal:
    def __init__(self, box, N_max=-1, k_max=1.0/0.1):
        """

        k_max should be in crystallographic inverse length (no 2*pi factor)
        """
        assert N_max == -1 or N_max > 30

        self.k_max = k_max
        self.N_max = N_max
        self.A = box.copy()
        # B is the "crystallographic" reciprocal vectors
        self.B = np.linalg.inv(self.A.transpose())

        k_mins = np.array([np.linalg.norm(b) for b in self.B])
        k_vol = np.prod(k_mins)
        self.k_mins = k_mins
        self.k_vol = k_vol

        if N_max == -1 or N_max > np.pi*k_max**3/(6*k_vol):
            # Do not deselect k-points
            self.k_prune = None
        else:
            # Use Cardano's formula to find k_prune
            p = -3.0*k_max**2/4
            q = 3.0*N_max*k_vol/np.pi - k_max**3/4
            D = (p/3)**3 + (q/2)**2
            #assert D < 0.0
            u = (-q/2+np.sqrt(D+0j))**(1.0/3)
            v = (-q/2-np.sqrt(D+0j))**(1.0/3)
            x = -(u+v)/2 - 1j*(u-v)*np.sqrt(3)/2
            self.k_prune = np.real(x) + k_max/2

        b1, b2, b3 = [(2*np.pi)*x.reshape((3,1,1,1)) for x in self.B]
        Nk1, Nk2, Nk3 = [np.ceil(k_max/dk) for dk in k_mins]
        kvals = \
            b1 * np.arange(Nk1, dtype=np.float64).reshape((1,Nk1,1,1)) + \
            b2 * np.arange(Nk2, dtype=np.float64).reshape((1,1,Nk2,1)) + \
            b3 * np.arange(Nk3, dtype=np.float64).reshape((1,1,1,Nk3))
        kvals = kvals.reshape((3, kvals.size/3))
        kdist = np.sqrt(np.sum(kvals**2, axis=0))*(1.0/(2*np.pi))
        I = np.nonzero(kdist<=k_max)[0]
        I = I[kdist[I].argsort()]
        kdist = kdist[I]
        kvals = kvals[:,I]
        if not self.k_prune is None:
            N = len(kdist)
            p = np.ones(N)
            # N(k) = a k^3
            # N'(k) = 3a k^2
            p[1:] = (self.k_prune/kdist[1:])**2
            I = np.nonzero(p > np.random.rand(N))[0]
            kdist = kdist[I]
            kvals = kvals[:,I]
        self.kvals = kvals
        self.kdist = kdist


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

    traj = XTC_reader(options.f)
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
    b1, b2, b3 = 2*np.pi*np.linalg.inv(reference_box.transpose())

    ## Temporary ugly...
    #b2 = b2 * (np.linalg.norm(b1)/np.linalg.norm(b2))
    #b3 = b3 * (np.linalg.norm(b1)/np.linalg.norm(b3))
    delta_k = np.linalg.norm(b1)
    Nk = options.nk
    max_k = delta_k*Nk
    if options.verbose:
        print('N = %i --> delta_x = %f [nm]' % (Nk, np.linalg.norm(a1)/Nk))

    b1 = b1.reshape((3,1,1,1))
    b2 = b2.reshape((3,1,1,1))
    b3 = b3.reshape((3,1,1,1))

    kvals = \
        b1 * np.arange(Nk, dtype=np.float64).reshape((1,Nk,1,1)) + \
        b2 * np.arange(Nk, dtype=np.float64).reshape((1,1,Nk,1)) + \
        b3 * np.arange(Nk, dtype=np.float64).reshape((1,1,1,Nk))
    kvals = kvals.reshape((3, kvals.size/3))
    kdist = np.sqrt(np.sum(kvals**2, axis=0))
    I = np.nonzero(kdist<=max_k)[0]
    I = I[kdist[I].argsort()]
    kdist = kdist[I]
    kvals = kvals[:,I]

    for N in [100,1000,10000,100000,1000000]:
        r = reciprocal(reference_box, N_max=N, k_max=10)
        print(r.k_prune, r.k_max, N, len(r.kdist))


    def add_rho_ks(frame):
        frame['rho_ks'] = [calc_rho_k(x, kvals) for x in frame['xs']]
        return frame

    traj = XTC_reader(options.f, 
                      max_frames=options.max_frames,
                      index_file=options.n)
    try:
        frame_list = cyclic_list([add_rho_ks(traj.next()) for x in range(N_tc)])
    except StopIteration:
        print('Failed to read %i frames (minimum required) from %s' % \
                  (N_tc, options.f))
        sys.exit(1)

    # * Assert box is not changed during consecutive frames
    # * Handle different time steps?
    # * Handle none cubic box in a good way
    # * Assert box is square

    Ntypes = len(traj.types)
    m = count(0)
    mij_list = [(m.next(),i,j) for i in range(Ntypes) for j in range(i,Ntypes)]
    type_combos = [traj.types[i]+'-'+traj.types[j] for _, i, j in mij_list]

    F_k_t_avs = [averager(np.zeros(len(kdist)), N_tc) for _ in mij_list]

    for frame_i, frame in enumerate(traj):
        frame_list[frame_i+N_tc-1] = add_rho_ks(frame)
        if options.verbose: print(frame['step'])

        rho_k_0 = frame_list[frame_i]['rho_ks']
        for time_i in range(N_tc):
            rho_k_i = frame_list[frame_i+time_i]['rho_ks']
            for m, i, j in mij_list:
                F_k_t_avs[m].add(np.real(rho_k_0[i]*rho_k_i[j].conjugate()), time_i)
                
    F_k_t_full = [np.array([F_k_t_avs[m].get_av(time_i) for time_i in range(N_tc)])
                  for m, _, _ in mij_list]

    # dirty smooth it out
    pts = 2*Nk
    rng = (-delta_k/4, max_k+delta_k/4) 
    Npoints, edges = np.histogram(kdist, bins=pts, range=rng)
    F_k_t = [np.zeros((N_tc, pts)) for _ in mij_list]
    F_k_t_sd = [np.zeros((N_tc, pts)) for _ in mij_list]
    k = 0.5*(edges[1:]+edges[:-1])
    t = delta_t*np.arange(N_tc)
    
    for m,_,_ in mij_list:
        ci = 0
        for i, n in enumerate(Npoints):
            if n == 0:
                F_k_t[m][:,i] = np.NaN
                continue
            s = F_k_t_full[m][:,ci:ci+n]
            F_k_t[m][:,i] = np.mean(s, axis=1)
            F_k_t_sd[m][:,i] = np.std(s, axis=1)
            ci += n
