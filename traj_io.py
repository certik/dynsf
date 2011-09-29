
__all__ = ['trajectory_iterator', 'XTC_reader', 'TRJ_reader', 'read_ndx_file']

from ctypes import cdll, byref, c_int, c_float, POINTER
from ctypes.util import find_library
from itertools import islice

import numpy as np
import re
import sys

class curry:
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.pending = args[:]
        self.kwargs = kwargs.copy()

    def __call__(self, *args, **kwargs):
        if kwargs and self.kwargs:
            kw = self.kwargs.copy()
            kw.update(kwargs)
        else:
            kw = kwargs or self.kwargs

        return self.fun(*(self.pending + args), **kw)

def trajectory_iterator(filename, index_file=None, step=1, max_frames=0):
        if filename.endswith('.xtc'):
            reader = XTC_reader
        elif re.match(r'^.+\.trj(\.(gz|bz2))?$', filename):
            reader = curry(TRJ_reader, x_factor=0.1, t_factor=1.0)
        else:
            raise RuntimeError('Unknown file format (suffix)')

        i = reader(filename, index_file=index_file)

        assert step > 0
        assert max_frames >= 0
        if max_frames == 0: 
            max_frames = sys.maxint
        elif step > 1:
            max_frames = max_frames*step
        i = islice(i, 0, max_frames, step)

        return i


lname = find_library('gmx')
libgmx = lname and cdll.LoadLibrary(lname)
np_ndp = np.ctypeslib.ndpointer
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
        np_ndp(dtype=xtcfloat_np, shape=(3,3), 
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
        np_ndp(dtype=xtcfloat_np, shape=(3,3), 
                               flags='f_contiguous, aligned'),
        np_ndp(dtype=xtcfloat_np, ndim=2, 
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
    def __init__(self, filename, index_file=None):
        if libgmx is None:
            raise RuntimeError("No libgmx found, can't read xtc-file")
        
        self._fio = libgmx.open_xtc(filename, 'r')
        if not self._fio:
            raise IOError("Failed to open file %s (for some reason)" % filename)

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
        if not self._open:
            raise StopIteration
        if not self._first_called:
            self._get_first()
        else:
            if not self._get_next():
                self.close()
                raise StopIteration
            
        xs = [self._x[:,I] for I in self.indexes]
        return {'N' : self._natoms.value,
                'types' : tuple(self.types),
                'box' : self._box.copy('F'),
                'step' : self._step.value,
                'time' : self._time.value,
                'xs' : xs,
                }



class TRJ_reader:
    """Read LAMMPS trajectory file, naive implementation
    """
    def __init__(self, filename, index_file=None, 
                 x_factor=1.0, t_factor=1.0):
        if filename.endswith('.gz'):
            from gzip import GzipFile as CFile
        elif filename.endswith('.bz2'):
            from bz2 import BZ2File as CFile
        else:
            CFile = open
        self._fh = CFile(filename, 'r')
        self._open = True
        self.index_file = index_file
        self._item_re = \
            re.compile(r'^ITEM: (TIMESTEP|NUMBER OF ATOMS|BOX BOUNDS|ATOMS) ?(.*)$')
        self.x_factor = x_factor
        self.t_factor = t_factor
        self.v_factor = x_factor/t_factor
        self._first_called = False
                

    def _setup_indexes(self):
        # This requires the knowledge of the number of atoms, N.
        # Hence, it cannot be called until after the first frame is read.
        N = self._natoms
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

    # ITEM: TIMESTEP
    # 81000
    # ITEM: NUMBER OF ATOMS
    # 1536
    # ITEM: BOX BOUNDS pp pp pp
    # 1.54223 26.5378
    # 1.54223 26.5378
    # 1.54223 26.5378
    # ITEM: ATOMS id type x y z vx vy vz 
    # 247 1 3.69544 2.56202 3.27701 0.00433856 -0.00099307 -0.00486166 
    # 249 2 3.73324 3.05962 4.14359 0.00346029 0.00332502 -0.00731005 
    # 463 1 3.5465 4.12841 5.34888 0.000523332 0.00145597 -0.00418675 

    def _read_frame_header(self):
        while True:
            L = self._fh.readline()
            m = self._item_re.match(L)
            if not m:
                if L == '':
                    self._fh.close()
                    self._open = False
                    raise StopIteration
                if L.strip() == '':
                    continue
                raise IOError("Failed to read/parse TRJ frame header")
            if m.group(1) == "TIMESTEP":
                step = int(self._fh.readline())
            elif m.group(1) == "NUMBER OF ATOMS":
                natoms = int(self._fh.readline())
            elif m.group(1) == "BOX BOUNDS":
                bbounds = [map(float, self._fh.readline().split())
                           for _ in range(3)]
                x = np.array(bbounds)
                box = np.diag(x[:,1]-x[:,0])
                if x.shape == (3,3):
                    box[1,0] = x[0,2]
                    box[2,0] = x[1,2]
                    box[2,1] = x[2,2]
                elif x.shape != (3,2):
                    raise IOError('Malformed box bounds in TRJ frame header')
            elif m.group(1) == "ATOMS":
                cols = tuple(m.group(2).split())
                # At this point, there should be only atomic data left
                return (step, natoms, box, cols)
            
    def _get_first(self):
        # Read first frame, update state of self, create indexes etc
        step, N, box, cols = self._read_frame_header()
        self._natoms = N
        self._step = step
        self._cols = cols        
        self._box = box

        def _all_in_cols(keys):
            for k in keys:
                if not k in cols:
                    return False
            return True

        if _all_in_cols(('id','x','y','z')):
            self._x_I = np.array(map(cols.index, ('x','y','z')))
        elif _all_in_cols(('id','xu','yu','zu')):
            self._x_I = np.array(map(cols.index, ('xu','yu','zu')))
        else:
            raise RuntimeError('TRJ file must contain at least atom-id, x, y, '
                               'and z coordinates to be useful.')
        self._id_I = cols.index('id')

        if _all_in_cols(('vx','vy','vz')):
            self._v_I = np.array(map(cols.index, ('vx','vy','vz')))            
        else:
            self._v_I = None

        #if 'type' in cols:
        #    self._type_I = cols.index('type')
        #else:
        #    self._type_I = None

        data = np.array([map(float, self._fh.readline().split()) 
                         for _ in range(N)])
        I = np.asarray(data[:,self._id_I], dtype=np.int)-1
        self._x = np.zeros((3,N), order='F')
        self._x[:,I] = data[:,self._x_I].transpose()
        if not self._v_I is None:
            self._v = np.zeros((3,N), order='F')
            self._v[:,I] = data[:,self._v_I].transpose()

        self._setup_indexes()

    
    def _get_next(self):
        # get next frame, update state of self
        step, N, box, cols = self._read_frame_header()
        assert(self._natoms == N)
        assert(self._cols == cols)
        self._step = step
        self._box = box
        
        data = np.array([map(float, self._fh.readline().split()) 
                         for _ in range(N)])
        I = np.asarray(data[:,self._id_I], dtype=np.int)-1
        self._x[:,I] = data[:,self._x_I].transpose()
        if not self._v_I is None:
            self._v[:,I] = data[:,self._v_I].transpose()
            
    def __iter__(self):
        return self
    
    def close(self):
        pass

    def next(self):
        if not self._open:
            raise StopIteration

        if self._first_called:
            self._get_next()
        else:
            self._get_first()
            
        xs = [self.x_factor*self._x[:,I] for I in self.indexes]
        res = {'N' : int(self._natoms),
               'types' : tuple(self.types),
               'box' : self.x_factor*self._box.copy('F'),
               'step' : int(self._step),
               'time' : self.t_factor*self._step,
               'xs' : xs }
        if not self._v_I is None:
            res['vs'] = [self.v_factor*self._v[:,I] for I in self.indexes]

        return res




def read_ndx_file(filename):
    """Read an ini-style gromacs index file
    
    Reads and parses named index file, returns a list
    of name-arrays-tuples, containing 
    name and indexes of the specified sections.
    """
    section_re = re.compile(r'^ *\[ *([a-zA-Z0-9_.-]+) *\] *$')
    sections = []
    members = []
    name = None
    with open(filename, 'r') as f:
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

