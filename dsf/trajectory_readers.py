
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

__all__ = ['XTC_reader', 'TRJ_reader', 'molfile_reader',
           'trajectory_readers']

from numpy import pi, sin, cos, arange, array, zeros
import numpy as np
import re
import sys

from itertools import count

from ctypes import cdll, byref, c_int, c_float, c_char_p, POINTER
from ctypes.util import find_library

from dsf.molfile_plugin import (
    MolfilePlugin,
    TRAJECTORY_PLUGIN_MAPPING, MOLFILE_PLUGIN_DIR,
    molfile_timestep_metadata_t, molfile_timestep_t,
    molfile_atom_t
    )


class abstract_trajectory_reader:
    """Provide a way to iterate through a MD-trajectory file, one frame at a time.

    Each frame is returned as a dictionary.
    {
     'index' : trajectory index,
     'box'   : simulation box as 3 row vectors (nm),
     'N'     : number of atoms,
     'x'     : particle positions as 3xN array (nm),
     'v'     : (*) particle velocities as 3xN array (nm/ps),
     'time'  : (*) simulation time (ps),
    }
    (*) may not be available, depends on reader and trajectory file format.
    """

    @classmethod
    def reader_available(cls):
        """Is this reader available on this system?

        E.g. are necessary 3rd party libraries available?
        """
        raise NotImplementedError:

    def __iter__(self):
        return self

    def next(self):
        """Get next trajectory frame
        """
        raise NotImplementedError:

    def close(self):
        """Close files, be done
        """
        raise NotImplementedError:

#
# L I B G M X
#
# libgmx comes with Gromacs and contain, among other things,
# functionality for reading xtc-files.
#

libgmx_name = find_library('gmx')
libgmx = libgmx_name and cdll.LoadLibrary(libgmx_name)
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


class XTC_reader(abstract_trajectory_reader):

    @classmethod
    def reader_available(cls):
        return libgmx is not None

    def __init__(self, filename):
        if libgmx is None:
            raise RuntimeError("No libgmx found, can't use XTC_reader!")

        self._fio = libgmx.open_xtc(filename, 'rb')
        if not self._fio:
            raise IOError("XTC_reader: Failed to open file %s" % filename)

        self._index = count(1)
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

    def _get_first(self):
        # Read first frame and update state of self accordingly
        _xfirst = POINTER(xtcfloat_ct)()
        res = libgmx.read_first_xtc(self._fio, self._natoms,
                                    self._step, self._time,
                                    self._box, _xfirst,
                                    self._prec, self._bOK)
        self._first_called = True
        if not res:
            raise IOError("XTC_reader: read_first_xtc failed")
        if not self._bOK.value:
            raise IOError("XTC_reader: corrupt frame in xtc-file?")

        N = self._natoms.value
        self._x = np.require(array(_xfirst[0:3*N]).reshape((3,N), order='F'),
                             xtcfloat_np, ['F_CONTIGUOUS', 'ALIGNED'])
        self._x.flags.writeable = False

    def _get_next(self):
        # get next frame, update state of self
        res = libgmx.read_next_xtc(self._fio, self._natoms.value,
                                   self._step, self._time,
                                   self._box, self._x,
                                   self._prec, self._bOK)
        if not res:
            return False
        if not self._bOK.value:
            raise IOError("XTC_reader: corrupt frame in xtc-file?")
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

        if self._first_called:
            if not self._get_next():
                self.close()
                raise StopIteration
        else:
           self._get_first()

        return dict(
            index = self._index.next()
            box = self._box.copy('F'),
            time = self._time.value,
            N = self._natoms.value,
            x = self._x,
            )


class TRJ_reader(abstract_trajectory_reader):
    """Read LAMMPS trajectory file

    This is a naive (and comparatively slow) implementation,
    written entirely in python.
    """

    @classmethod
    def reader_available(cls):
        return True

    def __init__(self, filename, x_factor=0.1, t_factor=1.0):
        if filename.endswith('.gz'):
            from gzip import GzipFile
            self._fh = GzipFile(filename, 'r')
        elif filename.endswith('.bz2'):
            from bz2 import BZ2File
            self._fh = BZ2File(filename, 'r')
        else:
            self._fh = open(filename,'r')

        self._open = True
        self._item_re = \
            re.compile(r'^ITEM: (TIMESTEP|NUMBER OF ATOMS|BOX BOUNDS|ATOMS) ?(.*)$')
        self.x_factor = x_factor
        self.t_factor = t_factor
        self.v_factor = x_factor/t_factor
        self._first_called = False
        self._index = count(1)

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
                raise IOError("TRJ_reader: Failed to read/parse TRJ frame header")
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
                    raise IOError('TRJ_reader: Malformed box bounds in TRJ frame header')
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
        if self._v_I is not None:
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
        if self._v_I is not None:
            self._v[:,I] = data[:,self._v_I].transpose()

    def __iter__(self):
        return self

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def next(self):
        if not self._open:
            raise StopIteration

        if self._first_called:
            self._get_next()
        else:
            self._get_first()

        xs = [self.x_factor*self._x[:,I] for I in self.indexes]
        res = dict(
            index = self._index.next(),
            N = int(self._natoms),
            box = self.x_factor*self._box.copy('F'),
            time = self.t_factor*self._step,
            x = self.x_factor*self._x,
            )

        if self._v_I is not None:
            res['v'] = self.v_factor*self._v

        return res


# M O L F I L E    P L U G I N
#

# Molfile plugin is using single precission floats
molfile_float_np = np.float32
molfile_float_ct = c_float

class molfile_reader(abstract_trajectory_reader):
    """Read a trajectory using the molfile_plugin package

    molfile_plugin is a part of VMD, and consists of
    plugins for a fairly large number of different trajectory
    formats (see molfile_plugin.TRAJECTORY_PLUGIN_MAPPING).

    filename - string, filename of trajectory file.
    index_file - string, filename of ini-style index file.
    plugin - string, name of plugin to use. If None, guess
             pluginname by looking at filename suffix.
    """

    @classmethod
    def reader_available(cls):
        return MOLFILE_PLUGIN_DIR is not None

    @classmethod
    def suggest_plugin(cls, filename_suffix):
        for _,_,sfx,plugin in TRAJECTORY_PLUGIN_MAPPING:
            if sfx == filename_suffix:
                return plugin
        return None

    def __init__(self, filename, plugin=None, x_factor=0.1, t_factor=1.0):

        if plugin is None:
            suffix = filename.rsplit('.', 1)[-1]
            plugin = self.suggest_plugin(suffix)

        if plugin is None:
            raise RuntimeError('molfile_reader: no suitable plugin known for file %s' % filename)

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
            raise RuntimeError('molfile_reader: failed to open file %s with plugin %s.' % (
                    filename, plugin))
        N = self._N.value

        if p.read_structure:
            # for e.g. lammpsplugin, read_structure needs to be called first so
            # that molfile_reader knows (internally) which coordinates and
            # velocites to map to which atoms.
            self._atoms_arr = (molfile_atom_t*N)()
            self._optflags = c_int()
            rc = p.read_structure(self._fh, byref(self._optflags), self._atoms_arr)
            if rc:
                raise IOError('molfile_reader: read structure failed for '
                              'file %s (plugin %s, rc %i)' % (filename, plugin, rc))
        else:
            self._atoms_arr = None

        self._v = None
        self._x = np.require(zeros((3,N)), molfile_float_np,
                             ['F_CONTIGUOUS', 'ALIGNED'])
        self._x.flags.writeable = False

        if p.read_timestep_metadata:
            # It seems only lammpsplugin offers this (but other formats could
            # include velocity information. how to test for that!?)
            tsm = self._timestep_metadata = molfile_timestep_metadata_t()
            rc = p.read_timestep_metadata(self._fh, byref(tsm))
            if rc:
                raise IOError('molfile_reader: read timestep metadata failed for '
                              'file %s (plugin %s, rc %i)' % (filename, plugin, rc))

            if tsm.has_velocities:
                self._v = np.require(zeros((3,N)), molfile_float_np,
                                     ['F_CONTIGUOUS', 'ALIGNED'])
                self._v.flags.writeable = False
        else:
            self._timestep_metadata = None

        # Now, set up the timestep structure
        self._ts = molfile_timestep_t()
        self._ts.coords = self._x.ctypes.data_as(POINTER(molfile_float_ct))
        if self._v is not None:
            self._ts.velocities = self._v.ctypes.data_as(POINTER(molfile_float_ct))
        else:
            # Set velocities to a NULL pointer
            self._ts.velocities = POINTER(molfile_float_ct)()

        # Set frame counter
        self._index = count(1)

    def __iter__(self):
        return self

    def next(self):
        if not self._mfp.plugin.read_next_timestep:
            raise StopIteration

        ts = self._ts
        rc = self._mfp.plugin.read_next_timestep(self._fh, self._N, byref(ts))
        if rc:
            self._mfp.close()
            raise StopIteration

        res = dict(
                   index = self._index.next(),
                   box = to_box(ts.A, ts.B, ts.C,
                                ts.alpha, ts.beta, ts.gamma)*self.x_factor,
                   N = self._N.value,
                   time = ts.physical_time*self.t_factor,
                   x = self._x*self.x_factor
                   )
        if self._v is not None:
            res['v'] = self._v[:,I]*self.v_factor

        return res

    def close(self):
        self._mfp.close()

def to_box(A, B, C, a, b, g):
    # Helper function that creates box vectors out of molfile-info
    f = pi/180.0
    return array(((A,           0.0,        0.0),
                  (B*cos(f*g),  B,          0.0),
                  (C*cos(f*b),  C*cos(f*a), C)))


trajectory_readers = (molfile_reader, XTC_reader, TRJ_reader)
