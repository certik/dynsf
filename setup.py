#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

setup(name = 'python-dynsf',
      version = '0.1',
      description = 'Tool for calculating the dynamical structure factor',
      author = 'Mattias Slabanja',
      author_email = 'slabanja@chalmers.se',
      packages = ['dsf'],
      ext_modules = [Extension('_rho_j_k', ['_rho_j_k.c'],
                               extra_compile_args='-fopenmp',
                               extra_link_args='-fopenmp')],
      scripts = ['dynsf'],
      requires = ['numpy']
      license      = "GPL2+",
      classifiers  = ['Development Status :: 4 - Beta',
                      'Intended Audience :: Science/Research',
                      'License :: OSI Approved :: GNU General Public License (GPL)',
                      'Programming Language :: Python',
                      ],

      )
