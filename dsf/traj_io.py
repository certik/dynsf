
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

__all__ = ['get_itraj', 'iwindow', 'read_ndx_file',
           'XTC_reader', 'TRJ_reader', 'molfile_reader']

from numpy import pi, sin, cos, arange, array, zeros
import numpy as np
import re
import sys

from itertools import islice, imap, count
from os.path import isfile
from collections import deque

from ctypes import cdll, byref, c_int, c_float, c_char_p, POINTER
from ctypes.util import find_library

from dsf.molfile_plugin import MolfilePlugin, \
    TRAJECTORY_PLUGIN_MAPPING, MOLFILE_PLUGIN_DIR, \
    molfile_timestep_metadata_t, molfile_timestep_t, \
    molfile_atom_t


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

def get_itraj(filename, step=1, max_frames=0, index_file=None,
              plugin=None):
    """Return a dynsf-style trajectory iterator

    step: (1 by default = every single frame), must be > 0.
    max_frames: (0 by default = no limit), must be >= 0.
    index_file, optional: Is used to explicitly split the
        particles/atoms in the trajectory into different
        categories.
    plugin, options: Explicitly specify molfile-plugin to use.
        Only used if the molfile_reader-plugins are found.
        If None, choose plugin based on filename suffix.

    Each iterator step consists of a dictionary containing keys:
    'N' : Total number of particles
    'xs' : List of coordinate arrays, one array for each
           particle/atom category.
    'box' :
    'time' :
    'step :

    Optional keys, depending on input:
    'vs' : List of coordinate velocites, one array for each
           particle/atom category.


    """

    if MOLFILE_PLUGIN_DIR:
        # Apparently we have a bunch of molfileplugins, just pass
        # it the filename and hope it works...
        # It seems molfile_reader defaults to Angstrom
        reader = curry(molfile_reader,
                       plugin=plugin, x_factor=0.1, t_factor=1.0)
    elif filename.endswith('.xtc') and libgmx:
        # libgmx is possibly faster than molfile_reader, but of course
        # less versatile.
        reader = XTC_reader
    elif re.match(r'^.+\.lammpstrj(\.(gz|bz2))?$', filename):
        # Fallback, only for lammpstrj-files
        reader = curry(TRJ_reader, x_factor=0.1, t_factor=1.0)
    else:
        raise RuntimeError('Unknown file format or no plugins found')

    if not isfile(filename):
        raise RuntimeError('File "%s" does not exist'%filename)

    i = reader(filename, index_file=index_file)

    assert step > 0
    assert max_frames >= 0
    if max_frames == 0:
        max_frames = sys.maxint
    elif step > 1:
        max_frames = max_frames*step
    i = islice(i, 0, max_frames, step)

    return i


def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # From the python.org
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)


class iwindow:
    """Sliding window iterator

    Returns consecutive windows (a windows is represented as a list
    of objects), created from an input iterator.

    Variable width (length of window, default 2),
    and stride (distance between the start of two consecutive
    window frames, default 1).
    Optional map_item to process each non-discarded object.
    Useful if stride > width and map_item is expensive (as compared to
    directly passing imap(fun, itraj) as itraj).
    If stride < width, you could as well directly pass "imap(fun, itraj)"
    """
    def __init__(self, itraj, width=2, stride=1, map_fun=None):
        self._raw_it = itraj
        if map_fun:
            self._it = imap(map_fun, self._raw_it)
        else:
            self._it = self._raw_it
        assert(stride >= 1)
        assert(width >= 1)
        self.width = width
        self.stride = stride
        self._window = None

    def __iter__(self):
        return self

    def next(self):
        if self._window is None:
            self._window = deque(islice(self._it, self.width), self.width)
        else:
            if self.stride >= self.width:
                self._window.clear()
                consume(self._raw_it, self.stride-self.width)
            else:
                for _ in xrange(min((self.stride, len(self._window)))):
                    self._window.popleft()
            for f in islice(self._it, min((self.stride, self.width))):
                self._window.append(f)

        if len(self._window) == 0:
            raise StopIteration

        return list(self._window)



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
    libgmx.open_xtc.argtypes = [c_char_p, c_char_p]

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

    Iterate through the frames of an xtc-file directly using libgmx.

    Each frame is represented as a dictionary.
    {'N': number of atoms,
     'box': simulation box as 3 row vectors (nm),
     'xs': xyz data as 3xN array (nm),
     'step': simulation step,
     'time': simulation time (ps) }
    """
    def __init__(self, filename, index_file=None):
        if libgmx is None:
            raise RuntimeError("No libgmx found, can't read xtc-file")

        self._fio = libgmx.open_xtc(filename, 'rb')
        if not self._fio:
            raise IOError("Failed to open file %s (for some reason)" % filename)

        self._natoms = xtcint_ct()
        self._step =   xtcint_ct()
        self._time =   xtcfloat_ct()
        self._box =    np.require(zeros((3,3)),
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
            self.indexes = [arange(N)]

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
        self._x = np.require(array(_xfirst[0:3*N]).reshape((3,N), order='F'),
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
        return dict(N = self._natoms.value,
                    types = tuple(self.types),
                    box = self._box.copy('F'),
                    step = self._step.value,
                    time = self._time.value,
                    xs = xs,
                    )



class TRJ_reader:
    """Read LAMMPS trajectory file

    This is a naive (and comparatively slow) implementation,
    written entirely in python.
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
            self.indexes = [arange(N)]

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
                x = array(bbounds)
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

        if _all_in_cols(('id','xu','yu','zu')):
            self._x_I = array(map(cols.index, ('xu','yu','zu')))
        elif _all_in_cols(('id','x','y','z')):
            self._x_I = array(map(cols.index, ('x','y','z')))
        else:
            raise RuntimeError('TRJ file must contain at least atom-id, x, y, '
                               'and z coordinates to be useful.')
        self._id_I = cols.index('id')

        if _all_in_cols(('vx','vy','vz')):
            self._v_I = array(map(cols.index, ('vx','vy','vz')))
        else:
            self._v_I = None

        if 'type' in cols:
            self._type_I = cols.index('type')
        else:
            self._type_I = None

        data = array([map(float, self._fh.readline().split())
                         for _ in range(N)])
        I = np.asarray(data[:,self._id_I], dtype=np.int)
        # Unless dump is done for group "all" ...
        I[np.argsort(I)] = arange(len(I))
        self._x = zeros((3,N), order='F')
        self._x[:,I] = data[:,self._x_I].transpose()
        if not self._v_I is None:
            self._v = zeros((3,N), order='F')
            self._v[:,I] = data[:,self._v_I].transpose()

        self._setup_indexes()


    def _get_next(self):
        # get next frame, update state of self
        step, N, box, cols = self._read_frame_header()
        assert(self._natoms == N)
        assert(self._cols == cols)
        self._step = step
        self._box = box

        data = array([map(float, self._fh.readline().split())
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
        res = dict(N = int(self._natoms),
                   types = tuple(self.types),
                   box = self.x_factor*self._box.copy('F'),
                   time = self.t_factor*self._step,
                   step = int(self._step),
                   xs = xs,
                   )
        if self._v_I is not None:
            res['vs'] = [self.v_factor*self._v[:,I] for I in self.indexes]

        return res


# Molfile plugin is using single precission floats
molfile_float_np = np.float32
molfile_float_ct = c_float

class molfile_reader:
    """Read a trajectory using the molfile_plugin package

    molfile_plugin is a part of VMD, and consists of
    plugins for a fairly large number of different trajectory
    formats (see molfile_plugin.TRAJECTORY_PLUGIN_MAPPING).

    filename - string, filename of trajectory file.
    index_file - string, filename of ini-style index file.
    plugin - string, name of plugin to use. If None, guess
             pluginname by looking at filename suffix.
    """
    def __init__(self, filename, index_file=None, plugin=None,
                 x_factor=1.0, t_factor=1.0):
        if plugin is None:
            suffix = filename.rsplit('.', 1)[-1]
            for _,_,sfx,plg in TRAJECTORY_PLUGIN_MAPPING:
                if sfx == suffix:
                    plugin = plg
                    break
        if plugin is None:
            raise RuntimeError('No suitable plugin known for file %s' % filename)

        self.x_factor = x_factor
        self.t_factor = t_factor
        self.v_factor = x_factor/t_factor

        self._N = c_int()
        suffix = filename.rsplit('.',1)[-1]

        self._mfp = MolfilePlugin(plugin)
        p = self._mfp.plugin

        self._fh = p.open_file_read(filename, suffix,
                                    byref(self._N))
        if not self._fh:
            raise RuntimeError('Failed to open file %s with plugin %s.' % \
                                   (filename, plugin))
        N = self._N.value

        if p.read_structure:
            # for e.g. lammpsplugin, read_structure needs to be called first so
            # that molfile_reader knows (internally) which coordinates and
            # velocites to map to which atoms.
            self._atoms_arr = (molfile_atom_t*N)()
            self._optflags = c_int()
            rc = p.read_structure(self._fh, byref(self._optflags), self._atoms_arr)
            if rc:
                raise IOError('Read structure failed for '
                              'file %s (plugin %s, rc %i)' % (filename, plugin, rc))
        else:
            self._atoms_arr = None

        self._v = None
        self._x = np.require(zeros((3,N)), molfile_float_np,
                             ['F_CONTIGUOUS', 'ALIGNED'])

        if p.read_timestep_metadata:
            # It seems only lammpsplugin offers this (but other formats could
            # include velocity information. how to test for that!?)
            tsm = self._timestep_metadata = molfile_timestep_metadata_t()
            rc = p.read_timestep_metadata(self._fh, byref(tsm))
            if rc:
                raise IOError('Read timestep metadata failed for '
                              'file %s (plugin %s, rc %i)' % (filename, plugin, rc))

            if tsm.has_velocities:
                self._v = np.require(zeros((3,N)), molfile_float_np,
                                     ['F_CONTIGUOUS', 'ALIGNED'])
        else:
            self._timestep_metadata = None

        # Now, set up the timestep structure
        self._ts = molfile_timestep_t()
        self._ts.coords = self._x.ctypes.data_as(POINTER(molfile_float_ct))
        if self._v is None:
            # Set velocities to a NULL pointer
            self._ts.velocities = POINTER(molfile_float_ct)()
        else:
            self._ts.velocities = self._v.ctypes.data_as(POINTER(molfile_float_ct))

        # Set frame counter
        self._fcnt = count(1)

        # Since we know N, let's also read the index_file already
        if index_file:
            self.types = []
            self.indexes = []
            for t, I in read_ndx_file(index_file):
                if I[0]<0 or I[-1]>=N:
                    raise RuntimeError('Invalid index found in index file')
                self.types.append(t)
                self.indexes.append(I)
        else:
            self.types = ['all']
            self.indexes = [arange(N)]

    def __iter__(self):
        return self

    def next(self):
        assert self._mfp.plugin.read_next_timestep

        N = self._N.value
        ts = self._ts
        rc = self._mfp.plugin.read_next_timestep(self._fh, N, byref(ts))
        if rc:
            self._mfp.close()
            raise StopIteration

        res = dict(N = N,
                   types = tuple(self.types),
                   box = to_box(ts.A, ts.B, ts.C,
                                ts.alpha, ts.beta, ts.gamma)*self.x_factor,
                   time = ts.physical_time*self.t_factor,
                   step = self._fcnt.next(),
                   xs = [self._x[:,I]*self.x_factor for I in self.indexes]
                   )
        if not self._v is None:
            res['vs'] = [self._v[:,I]*self.v_factor for I in self.indexes]

        return res


def to_box(A, B, C, a, b, g):
    # Helper function that creates box vectors out of molfile-info
    f = pi/180.0
    return array(((A,           0.0,        0.0),
                  (B*cos(f*g),  B,          0.0),
                  (C*cos(f*b),  C*cos(f*a), C)))




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
                    sections.append((name, np.unique(array(members))-1))
                name = m.group(1)
                members = []
            elif not L.isspace():
                members += map(int, L.split())
        if members and name:
            sections.append((name, np.unique(array(members))-1))
    return sections
