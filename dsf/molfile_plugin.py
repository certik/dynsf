#!/usr/bin/env python

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

__all__ = ['MolfilePlugin', 'PLUGIN_TRAJECTORY_MAPPING']

import os
from sysconfig import get_config_var

from ctypes import cdll, CDLL, RTLD_GLOBAL, \
    POINTER as PTR, CFUNCTYPE as CFT, \
    Structure, cast, pointer, byref, \
    c_int, c_uint, c_float, c_double, c_char, c_char_p, c_void_p
from ctypes.util import find_library

# Maybe a bit ugly. Is there a more kosher way of making sure
# that the libstdc++ symbols are available to the molfile-plugins?
_cxx = CDLL(find_library('stdc++'), mode=RTLD_GLOBAL)

c_int_p = PTR(c_int)
c_int_pp = PTR(c_int_p)
c_char_pp = PTR(c_char_p)
c_char_ppp = PTR(c_char_pp)
c_float_p = PTR(c_float)
c_float_pp = PTR(c_float_p)

#
# As of molfile_plugin abiversion 16, the following trajectory 
# formats are supported:
#
# Molecular Dynamics Trajectory File Plugins
#
#     AMBER 'binpos' trajectory reader (.binpos)
#     AMBER "CRD" trajectory reader (.crd, .crdbox)
#     AMBER NetCDF trajectory reader (.nc)
#     CHARMM, NAMD, X-PLOR "DCD" reader/writer (.dcd)
#     CPMD (CPMD trajectory) reader (.cpmd)
#     DLPOLY HISTORY file reader (.dlpolyhist)
#     Gromacs TRR/XTC reader (.trr, .xtc)
#     LAMMPS trajectory reader (.lammpstrj)
#     MMTK NetCDF trajectory reader (.nc)
#     VASP trajectories of ionic steps (.xml, .OUTCAR, .XCATCAR)
#     VTF trajectory files (.vtf)
#     XCrySDen, Quantum Espresso XSF/AXSF trajectory files (.axsf, .xsf)
#     XYZ trajectory files (.xyz) 
#

PLUGIN_TRAJECTORY_MAPPING = (
    #(SOFTWARE, TYPE, SUFFIX, PLUGIN)
    ('AMBER',   '"binpos"',       'binpos',   'binposplugin'),
    ('AMBER',   '"CRD"',          'crd',      'crdplugin'), 
    ('AMBER',   'NetCDF',         'nc',       'netcdfplugin'),
    ('CHARMM',  '"DCD" - CHARMM, NAMD, XPLOR',
                                  'dcd',      'dcdplugin'),
    ('CPMD',    'CPMD',           'cpmd',     'cpmdplugin'),
    ('DLPOLY',  'DLPOLY History', 'dlpolyhist','dlpolyplugin'),
    ('GROMACS', 'Gromacs XTC',    'xtc',      'gromacsplugin'),
    ('GROMACS', 'Gromacs TRR',    'trr',      'gromacsplugin'),
    ('LAMMPS',  'LAMMPS Trajectory','lammpstrj','lammpsplugin'),
    ('VASP',    'VASP ionic steps','xml',     'vaspxmlplugin'),
    ('VASP',    'VASP ionic steps','OUTCAR',  'vaspoutcarplugin'),
    ('VTF',     'VTF trajectory',  'vtf',     'vtfplugin'),
    ('XCrySDen','XSF trajectory',  'xsf',     'xsfplugin'),
    ('XCrySDen','AXSF trajectory', 'axsf',    'xsfplugin'),
    ('',        'XYZ trajectory',  'xyz',     'xyzplugin'))


PLUGIN_DIR = '/home/slabanja/opt/vmd/1.9.0/lib/plugins/LINUXAMD64/molfile'


MOLFILE_PLUGIN_TYPE = "mol file reader"
VMDPLUGIN_SUCCESS = 0
VMDPLUGIN_ERROR = -1

class vmdplugin_t(Structure):
    _fields_ = [('abiversion', c_int),
                ('type', c_char_p),
                ('name', c_char_p),
                ('prettyname', c_char_p),
                ('author', c_char_p),
                ('majorv', c_int),
                ('minorv', c_int),
                ('is_reentrant', c_int)]

class molfile_metadata_t(Structure):
    _fields_ = [('database', c_char*81),
                ('accession', c_char*81),
                ('date', c_char*81),
                ('title', c_char*81),
                ('remarklen', c_int),
                ('remarks', c_char_p)]

class molfile_atom_t(Structure):
    _fields_ = [('name', c_char*16),
                ('type', c_char*16),
                ('resname', c_char*16),
                ('resid', c_int),
                ('segid', c_char*16),
                ('chain', c_char*16),
                ('altloc', c_char*16),
                ('insertion', c_char*16),
                ('occupancy', c_float),
                ('bfactor', c_float),
                ('mass', c_float),
                ('charge', c_float),
                ('radius', c_float),
                ('atomicnumber', c_int)]

class molfile_timestep_metadata_t(Structure):
    _fields_ = [('count', c_uint),
                ('avg_bytes_per_timestamp', c_uint),
                ('has_velocities', c_int)]

class molfile_qm_metadata_t(Structure):
    pass

class molfile_qm_timestep_t(Structure):
    pass

class molfile_timestep_t(Structure):
    _fields_ = [('coords', PTR(c_float)),
                ('velocities', PTR(c_float)),
                ('A', c_float),
                ('B', c_float),
                ('C', c_float),
                ('alpha', c_float),
                ('beta', c_float),
                ('gamma', c_float),
                ('physical_time', c_double)]

class molfile_volumetric_t(Structure):
    _fields_ = [('dataname', c_char*256),
                ('origin', c_float*3),
                ('xaxis', c_float*3),
                ('yaxis', c_float*3),
                ('zaxis', c_float*3),
                ('xsize', c_int),
                ('ysize', c_int),
                ('zsize', c_int),
                ('has_color', c_int)]


dummy_fun_t = CFT(c_int)
class molfile_plugin_t(Structure):
    # ABI from molfile abiversion 16
    _fields_ = [('abiversion', c_int),
                ('type', c_char_p),
                ('name', c_char_p),
                ('prettyname', c_char_p),
                ('author', c_char_p),
                ('majorv', c_int),
                ('minorv', c_int),
                ('is_reentrant', c_int),
                ('filename_extension', c_char_p),
#
# void *(* open_file_read)(const char *filepath, const char *filetype, int *natoms);
                ('open_file_read', CFT(c_void_p, c_char_p, c_char_p, c_int_p)),
#                
# int (*read_structure)(void *, int *optflags, molfile_atom_t *atoms);
                ('read_structure', CFT(c_int, c_void_p, c_int_p,
                                       PTR(molfile_atom_t))),
#
# int (*read_bonds)(void *, int *nbonds, int **from, int **to, float **bondorder,
#                   int **bondtype, int *nbondtypes, char ***bondtypename);
                ('read_bonds', dummy_fun_t),
#
# int (* read_next_timestep)(void *, int natoms, molfile_timestep_t *);
                ('read_next_timestep', CFT(c_int, c_void_p, c_int,
                                           PTR(molfile_timestep_t))),
#
# void (* close_file_read)(void *);
                ('close_file_read', CFT(None, c_void_p)),
#
# void *(* open_file_write)(const char *filepath, const char *filetype,
#      int natoms);
                ('open_file_write', dummy_fun_t),
#
#  int (* write_structure)(void *, int optflags, const molfile_atom_t *atoms);
                ('write_structure', dummy_fun_t),
#
#  int (* write_timestep)(void *, const molfile_timestep_t *);
                ('write_timestep', dummy_fun_t),
#
#  void (* close_file_write)(void *);
                ('close_file_write', dummy_fun_t),
#
#  int (* read_volumetric_metadata)(void *, int *nsets,
#        molfile_volumetric_t **metadata);
                ('read_volumetric_metadata', CFT(c_int, c_void_p, c_int_p, 
                                                 PTR(PTR(molfile_volumetric_t)))),
#
#  int (* read_volumetric_data)(void *, int set, float *datablock,
#        float *colorblock);
                ('read_volumetric_data', CFT(c_int, c_void_p, c_int, c_float_p, 
                                             c_float_p)),
#
#  int (* read_rawgraphics)(void *, int *nelem, const molfile_graphics_t **data);
                ('read_rawgraphics', dummy_fun_t),
#
#  int (* read_molecule_metadata)(void *, molfile_metadata_t **metadata);
                ('read_molecule_metadata', CFT(c_int, c_void_p, 
                                               PTR(PTR(molfile_metadata_t)))),
#
#  int (* write_bonds)(void *, int nbonds, int *from, int *to, float *bondorder,
#                     int *bondtype, int nbondtypes, char **bondtypename);
                ('write_bonds', dummy_fun_t),                
#
#  int (* write_volumetric_data)(void *, molfile_volumetric_t *metadata,
#                                float *datablock, float *colorblock);
                ('write_volumetric_data', dummy_fun_t),
#
#  int (* read_angles)(void *handle, int *numangles, int **angles, int **angletypes,
#                      int *numangletypes, char ***angletypenames, int *numdihedrals,
#                      int **dihedrals, int **dihedraltypes, int *numdihedraltypes,
#                      char ***dihedraltypenames, int *numimpropers, int **impropers,
#                      int **impropertypes, int *numimpropertypes, char ***impropertypenames,
#                      int *numcterms, int **cterms, int *ctermcols, int *ctermrows);
                ('read_angles', dummy_fun_t),
#
#  int (* write_angles)(void *handle, int numangles, const int *angles, const int *angletypes,
#                       int numangletypes, const char **angletypenames, int numdihedrals,
#                       const int *dihedrals, const int *dihedraltypes, int numdihedraltypes,
#                       const char **dihedraltypenames, int numimpropers,
#                       const int *impropers, const int *impropertypes, int numimpropertypes,
#                       const char **impropertypenames, int numcterms, const int *cterms,
#                       int ctermcols, int ctermrows);
                ('write_angles', dummy_fun_t),
#
#  int (* read_qm_metadata)(void *, molfile_qm_metadata_t *metadata);
                ('read_qm_metadata', dummy_fun_t),
#
#  int (* read_qm_rundata)(void *, molfile_qm_t *qmdata);
                ('read_qm_rundata', dummy_fun_t),
#
#  int (* read_timestep)(void *, int natoms, molfile_timestep_t *,
#                        molfile_qm_metadata_t *, molfile_qm_timestep_t *);
                ('read_timestep', CFT(c_int, c_void_p, c_int, PTR(molfile_timestep_t),
                                      PTR(molfile_qm_metadata_t),
                                      PTR(molfile_qm_timestep_t))),
#
#  int (* read_timestep_metadata)(void *, molfile_timestep_metadata_t *);
                ('read_timestep_metadata', CFT(c_int, c_void_p,
                                               PTR(molfile_timestep_metadata_t))),
#
#  int (* read_qm_timestep_metadata)(void *, molfile_qm_timestep_metadata_t *);
                ('read_qm_timestep_metadata', dummy_fun_t),
#
#  int (* cons_fputs)(const int, const char*);
                ('cons_fputs', CFT(c_int, c_int, c_char_p))]




# typedef int (*vmdplugin_register_cb)(void *, vmdplugin_t *);
vmdplugin_register_cb_t = CFT(c_int, c_void_p, PTR(vmdplugin_t))


class MolfilePlugin:
    def __init__(self, plugin_name, plugin_dir=''):
        
        plugin_name += get_config_var('SO')
        fn = os.path.join(plugin_dir, plugin_name)
        lib = cdll.LoadLibrary(fn)
    
        # extern int vmdplugin_init(void);
        # extern int vmdplugin_fini(void);
        # extern int vmdplugin_register(void *, vmdplugin_register_cb);
        
        lib.vmdplugin_init.restype = c_int
        lib.vmdplugin_fini.restype = c_int
        lib.vmdplugin_register.restype = c_int    
        lib.vmdplugin_register.argtypes = (c_void_p, vmdplugin_register_cb_t)

        plugin_p = pointer(molfile_plugin_t(0))
        def py_reg_cb(v, p):
            pc = p.contents
            if pc.type == MOLFILE_PLUGIN_TYPE:
                if pc.abiversion >= 15:
                    plugin_p.contents = cast(p, PTR(molfile_plugin_t)).contents
            return VMDPLUGIN_SUCCESS

        vmdplugin_register_cb = vmdplugin_register_cb_t(py_reg_cb)
        if lib.vmdplugin_init() != VMDPLUGIN_SUCCESS:
            raise RuntimeError('Failed to init %s' % plugin_name)
        if lib.vmdplugin_register(None, vmdplugin_register_cb) != VMDPLUGIN_SUCCESS:
            raise RuntimeError('Failed to register %s' % plugin_name)

        self._lib = lib
        self.plugin = plugin_p.contents

    def close(self):
        self.plugin = molfile_plugin_t(0)
        self._lib.vmdplugin_fini()




#def test(filename, plugin):

#filename = '/home/slabanja/lammps/SPC-E-case/bild/dump_all.lammpstrj'
#plugin = 'lammpsplugin'

#filename = 'test.lammpstrj'
#plugin = 'lammpsplugin'

#filename = '/home/slabanja/lammps/SPC-E-case/T250-run/dump_all_10fs.xtc'
#plugin = 'gromacsplugin'


# mfile = MolfilePlugin(plugin)
# N = c_int()
# suffix = filename.rsplit('.',1)[-1]
# handle = mfile.plugin.open_file_read(filename, suffix, byref(N))
# if not handle:
#     print("Failed to open file")
#     import sys
#     sys.exit(1)
    
# if mfile.plugin.read_structure:
#     atoms = (molfile_atom_t*N.value)()
#     optflags = c_int()
#     rc = mfile.plugin.read_structure(handle, byref(optflags), atoms)
#     print rc

# if mfile.plugin.read_next_timestep:
#     ts = molfile_timestep_t()
#     ts.coords = (c_float*(3*N.value))()
#     ts.velocities = (c_float*(3*N.value))()

#     ts.coords[0]=3.141
#     ts.velocities[0]=1.234
#     rc = mfile.plugin.read_next_timestep(handle, N, byref(ts))
#     print rc
#     rc = mfile.plugin.read_next_timestep(handle, N, byref(ts))
#     print rc
# #    rc = mfile.plugin.read_next_timestep(handle, N, byref(ts))
# #    print rc
# #return ts  
  
if __name__ == '__main__':
    for _,_,ext,pn in PLUGIN_TRAJECTORY_MAPPING:
        p = MolfilePlugin(pn)
        print("%s  %s  %s" % (pn, ext, p.plugin.filename_extension)) #.filename_extension))
