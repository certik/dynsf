__all__ = ['filonc']

from numpy import sin, cos, zeros, arange, mod, sum

def filonc(f, dx, dk):
    """Calculate fourier cosine transform of function f(x) using Filons method

    If f is a 2D-array, the transformation is done along each individual
    column (axis=0).
    f(x) should have an odd number of elements.
    """
    
    orig_shape = f.shape
    
    if len(f.shape) == 1:
        f = f.reshape(f.shape+(1,1))
    elif len(f.shape) == 2:
        f = f.reshape(f.shape+(1,))
    else:
        raise RuntimeError('that many dimension are currently not supported')

    N = f.shape[0]
    Nmax = N-1
    if mod(Nmax, 2) != 0:
        raise RuntimeError('f should have an odd length')

    # Split into even (E) and odd (O) indexed parts
    fE = f[0::2,:,:]
    fO = f[1::2,:,:]

    # axis=3 represents the reciprocal dimension
    k = dk*arange(0.0, N).reshape((1,1,N))
    theta = dx*k

    # cos(theta*xi)
    cos_t_xi = cos(theta * arange(0.0, N).reshape((N,1,1)))
    cos_t_xi[0,:,:] *= 0.5
    cos_t_xi[Nmax,:,:] *= 0.5
    cos_t_xiE = cos_t_xi[0::2,:,:]
    cos_t_xiO = cos_t_xi[1::2,:,:]

    theta = theta[:,:,1:]
    sin_t = sin(theta)
    cos_t = cos(theta)
    sin2_t = sin_t*sin_t
    cos2_t = cos_t*cos_t
    theta2 = theta*theta
    itheta3 = 1.0/(theta2*theta)

    alpha = zeros((1,1,N))
    beta = zeros((1,1,N))
    gamma = zeros((1,1,N))
    beta[0,0,0] = 2.0/3.0
    gamma[0,0,0] = 4.0/3.0
    
    alpha[0,0,1:] = itheta3*(theta2 + theta*sin_t*cos_t - 2*sin2_t)
    beta[0,0,1:] = 2*itheta3*(theta*(1+cos2_t) - 2*sin_t*cos_t)
    gamma[0,0,1:] = 4*itheta3*(sin_t - theta*cos_t)

    F = 2*dx*(alpha*f[Nmax,:,:]*sin(k*f[Nmax,:,:]) +
              beta*sum(fE*cos_t_xiE, axis=0) +
              gamma*sum(fO*cos_t_xiO, axis=0))

    F = F.transpose(2,1,0).reshape(orig_shape)
    return F

