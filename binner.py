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
