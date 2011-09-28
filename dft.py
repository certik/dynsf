__all__ = ['fram', 'bak']

import numpy as np

def bak(F, q, r):
    F = np.array(F)
    k = np.array(q*2*np.pi)

    F[np.nonzero(np.isnan(F))] = 0.0

    r = np.array(r)
    r_ = r
    if r[0] == 0.0:
        r_ = r[1:]

    Nk = len(k)
    Nr_ = len(r_)

    dk = k[1]-k[0]
    k = k.reshape((Nk, 1))
    F = F.reshape((Nk, 1))
    
    r_ = r_.reshape((1, Nr_))

    S_ = dk*4*np.pi*np.sum(k*F*np.sin(-k*r_)/r_, axis=0)
    return (r_.reshape((Nr_,)), S_)
    


def fram():
    pass
