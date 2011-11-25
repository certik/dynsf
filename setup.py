#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars

from build_config import *

if local_compiler is not None:
    # Kludge: Force compiler of choice for building _rho_j_k.c.
    # _rho_j_k.c only contains a plain c-function with no python-
    # dependencies at all. Hence, just blatantly set simple 
    # compiler, linker and flags.
    # (inspired by GPAW setup.py)

    config_vars = get_config_vars()
    for key in ['BASECFLAGS', 'CFLAGS', 'OPT', 'PY_CFLAGS',
                'CCSHARED', 'CFLAGSFORSHARED', 'LINKFORSHARED',
                'LIBS', 'SHLIBS']:
        config_vars[key] = ''

    config_vars['CC'] = local_compiler
    config_vars['LDSHARED'] = ' '.join([local_linker] +
                                       local_link_shared)

rho_j_k_d_ext = Extension('dsf._rho_j_k_d', 
                          sources=['src/_rho_j_k.c'],
                          define_macros=[('RHOPREC', 'double')],
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args,
                          )

rho_j_k_s_ext = Extension('dsf._rho_j_k_s', 
                          sources=['src/_rho_j_k.c'],
                          define_macros=[('RHOPREC', 'float')],
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args,
                          )


setup(name = 'python-dynsf',
      version = '0.1.1+',
      description = 'Tool for calculating the dynamical structure factor',
      author = 'Mattias Slabanja',
      author_email = 'slabanja@chalmers.se',
      packages = ['dsf'],
      ext_modules = [rho_j_k_d_ext,
                     rho_j_k_s_ext],
      scripts = ['dynsf'],
      requires = ['numpy'],
      license      = "GPL2+",
      classifiers  = ['Development Status :: 3 - Alpha',
                      'Intended Audience :: Education', 
                      'Intended Audience :: Science/Research',
                      'License :: OSI Approved :: GNU General Public License (GPL)',
                      'Programming Language :: Python',
                      'Programming Language :: C',
                      'Topic :: Scientific/Engineering :: Chemistry',
                      'Topic :: Scientific/Engineering :: Physics'
                      ]
      )


