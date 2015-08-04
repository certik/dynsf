# Simple Chain of Particles #

Using LAMMPS to simulate a one dimensional (periodic boundary) chain of 100 particles, bonded together with a [Morse style](http://lammps.sandia.gov/doc/bond_morse.html) bond.

## LAMMPS input ##
The 100 particles, each having a mass of 24.3 u, were placed along the z-axis with a nearest neighbor distance of 3 Å in a 10x10x300 Å<sup>3</sup> simulation box. LAMMPS were setup according to
```
units metal
boundary p p p

atom_style bond
bond_style morse

read_data data.x
bond_coeff 1 1.0 1.0 3.0

velocity all create 3600 123123 units box
velocity all set 0.0 0.0 NULL units box

fix integrator all nve
timestep 1e-2
```

After equilibrating for 1000000 time steps, a consecutive 1000000 time steps were dumped into a XTC-file.

From a simply analysis of some 1000 of the first trajectory time steps, considering a subset of k-values corresponding to a line along the particle chain, one can obtain e.g. the static structure factor through the intermediate scattering function (S(k)=F(k,0)),
![http://wiki.dynsf.googlecode.com/git/images/S_k.png](http://wiki.dynsf.googlecode.com/git/images/S_k.png)

Or the dynamical structure factor, S(k,w), as a filled contour plot,
![http://wiki.dynsf.googlecode.com/git/images/S_k_w.png](http://wiki.dynsf.googlecode.com/git/images/S_k_w.png)


# Sodium Chloride at high-ish temperature #

Considering a simple NaCl system consisting of 475 + 475 ions, in a 31.6<sup>3</sup> Å<sup>3</sup> periodic boundary box, simulated at a temperature around 1200 K.

## LAMMPS input ##
The relevant LAMMPS input could be something in the line with
```
units metal
boundary p p p
atom_style charge

read_data data.x
group  Na type 1
group  Cl type 2

# Parameters from Lewis and Singer, J Chem Soc Faraday Trans, _71_, 41
pair_style born/coul/long 7.6 15
pair_coeff 1 1  0.26373  0.317  2.340  1.0487  -0.49938
pair_coeff 1 2  0.21099  0.317  2.755  6.6613  -8.6767
pair_coeff 2 2  0.15824  0.317  3.170  72.409  -145.44
pair_modify shift yes
kspace_style pppm 1e-5
```

A surf-plot of the partial S<sub>Na Na</sub>(k,w) using Matlab's surf plot,
![http://wiki.dynsf.googlecode.com/git/images/S_k_w_NaNa.png](http://wiki.dynsf.googlecode.com/git/images/S_k_w_NaNa.png)

