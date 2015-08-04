**Dynsf** is a tool for calculating the (partial) [dynamical structure factor](http://en.wikipedia.org/wiki/Dynamic_structure_factor) from a molecular dynamics (MD) simulation trajectory.

By calculating the dynamical structure factor, S(k,w), it is possible to directly compare a MD calculation to measurements from inelastic scattering experiments.

If available, dynsf can use the molfile\_plugin, a part of the Visual Molecular Dynamics ([VMD](http://www.ks.uiuc.edu/Research/vmd/)) software package, to read several different MD trajectory file formats. As alternatives to using molfile\_plugin, dynsf can also use libgmx ([Gromacs](http://www.gromacs.org)) to read Gromacs XTC-files, or it can directly parse [LAMMPS](http://lammps.sandia.gov/)-style trajectory dumps.

The dynamical structure factor is calculated via the intermediate scattering function, F(k,t), which in turn is calculated from direct time correlation calculation of the fourier transformed density, rho(**k**).

A few examples [can be found here](Examples.md).

Information about how to get started [can be found here](HowTo.md).

The program is currently in an early stage of its life (think alpha).