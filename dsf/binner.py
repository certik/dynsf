
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

__all__ = ['binner']

import numpy as np


def binner(y, x=None, points=30):
    if x is None:
        x = np.arange(len(y))

    delta = x[1]-x[0]
    max_x = x[-1]
    rng = (-0.5*delta, max_x+0.5*delta) 
    Npoints, edges = np.histogram(x, bins=points, range=rng)
    av = np.zeros(points)
    sd = np.zeros(points)

    xnew = 0.5*(edges[1:]+edges[:-1])
    
    ci = 0
    for i, n in enumerate(Npoints):
        if n == 0:
            av[i] = np.NaN
        else:
            s = y[ci:ci+n]
            av[i] = np.mean(s)
            sd[i] = np.std(s)
            ci += n
    return (av, sd, xnew)
