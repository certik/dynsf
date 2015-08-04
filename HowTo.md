# Building and installing #
## Quick start ##
Dynsf is a simple python-distutils program, hence, it's installable by simply unpacking the code, change directory into the code root, and doing something in the line with
```
./setup.py install --prefix=/my/install/prefix
```
Make sure the "binary" and the modules are found (example in bash; the "lib/pythonX.Y"-part should reflect the actual install location for the dynsf module files),
```
export PATH=$PATH:/my/install/prefix/bin
export PYTHONPATH=$PYTHONPATH:/my/install/prefix/lib/pythonX.Y/site-packages
```

It should then be possible to invoke dynsf from the command line.

### Using libgmx to read xtc-files ###
If libgmx (Gromacs shared library) is found, dynsf can use it to read xtc-files.
If Gromacs is not installed in a standard location on the system, the dynamical loader may need to be informed of where to look for shared libraries (on typical GNU/Linux systems, one can use the LD\_LIBRARY\_PATH environment variable, or on OS-X systems, the DYLD\_LIBRARY\_PATH environment variable can be used).

### Using the molfile-plugin ###
The molfile-plugin is part of VMD, and can be used to read various different trajectory file formats.
In order for dynsf to be able to use it, the vmd command line script need to be in the normal command path (the PATH environment variable). Note that the molfile plugins need to be compiler for the same architecture as the main python process. It is e.g. not possible to dynamically load "32-bit" x86 plugins if the python interpreter is a "64-bit" x86\_64 binary.

## Choice of compiler and options ##
A small part (the part calculating rho(**k**) for a set of atoms and **k**-points) is written in C.
By default, the system compiler will be used. This can be overridden by, prior to the build-phase, altering the  build\_config.py file accordingly.