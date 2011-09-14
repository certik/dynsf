#!/usr/bin/env python

from itertools import count, islice
from collections import deque
import re
import sys
import numpy as np

from traj_io import XTC_reader, TRJ_reader
from rho_j_k import calc_rho_k, calc_rho_j_k

def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # From the python.org
    if n is None:
        collections.deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)

def npopleft(deq, n):
    if len(deq) >= n:
        for _ in range(n):
            deq.popleft()
    else:
        deq.clear()

class curry:
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.pending = args[:]
        self.kwargs = kwargs.copy()

    def __call__(self, *args, **kwargs):
        if kwargs and self.kwargs:
            kw = self.kwargs.copy()
            kw.update(kwargs)
        else:
            kw = kwargs or self.kwargs

        return self.fun(*(self.pending + args), **kw)

class averager:
    def __init__(self, init_array, N_slots):
        assert type(N_slots) is int
        self._N = N_slots
        self._data = [np.array(init_array) for n in range(N_slots)]
        self._samples = [0]*N_slots
    def add(self, array, slot):
        self._data[slot] = self._data[slot] + array
        self._samples[slot] += 1
    def get_av(self, slot):
        n = self._samples[slot]
        assert n > 0
        f = 1.0/n
        return f*self._data[slot]

class reciprocal:
    def __init__(self, box, N_max=-1, q_max=1.0/0.1):
        """Create a suitable set of reciprocal coordinates

        Optionally limit the set to approximately N_max points by
        randomly removing points. The points are removed in such a way
        that for q>q_prune, the points will be radially uniformely 
        distributed (the value of q_prune is calculated from q_max, N_max,
        and the shape of the box). 

        q_max should be the "crystallographic reciprocal length" (no 2*pi factor)
        """
        assert(N_max == -1 or N_max > 1000)

        self.q_max = q_max
        self.N_max = N_max
        self.A = box.copy()
        # B is the "crystallographic" reciprocal vectors
        self.B = np.linalg.inv(self.A.transpose())

        q_mins = np.array([np.linalg.norm(b) for b in self.B])
        q_vol = np.prod(q_mins)
        self.q_mins = q_mins
        self.q_vol = q_vol

        if N_max == -1 or N_max > np.pi*q_max**3/(6*q_vol):
            # Do not deselect k-points
            self.q_prune = None
        else:
            # Use Cardano's formula to find q_prune
            p = -3.0*q_max**2/4
            q = 3.0*N_max*q_vol/np.pi - q_max**3/4
            D = (p/3)**3 + (q/2)**2
            #assert D < 0.0
            u = (-q/2+np.sqrt(D+0j))**(1.0/3)
            v = (-q/2-np.sqrt(D+0j))**(1.0/3)
            x = -(u+v)/2 - 1j*(u-v)*np.sqrt(3)/2
            self.q_prune = np.real(x) + q_max/2

        b1, b2, b3 = [(2*np.pi)*x.reshape((3,1,1,1)) for x in self.B]
        Nk1, Nk2, Nk3 = [np.ceil(q_max/dk) for dk in q_mins]
        kvals = \
            b1 * np.arange(Nk1, dtype=np.float64).reshape((1,Nk1,1,1)) + \
            b2 * np.arange(Nk2, dtype=np.float64).reshape((1,1,Nk2,1)) + \
            b3 * np.arange(Nk3, dtype=np.float64).reshape((1,1,1,Nk3))
        kvals = kvals.reshape((3, kvals.size/3))
        kdist = np.sqrt(np.sum(kvals**2, axis=0))*(1.0/(2*np.pi))
        I = np.nonzero(kdist<=q_max)[0]
        I = I[kdist[I].argsort()]
        kdist = kdist[I[1:]]
        kvals = kvals[:,I[1:]]
        if not self.q_prune is None:
            N = len(kdist)
            p = np.ones(N)
            # N(k) = a k^3
            # N'(k) = 3a k^2
            p = (self.q_prune/kdist)**2
            I = np.nonzero(p > np.random.rand(N))[0]
            kdist = kdist[I]
            kvals = kvals[:,I]
        self.kvals = kvals
        self.kdist = kdist


if __name__ == '__main__':
    import sys
    import optparse
    from math import ceil

    parser = optparse.OptionParser()
    parser.add_option('-f', '', metavar='TRAJECTORY_FILE',
                      help='Trajectory file in xtc or lammps-dump format')
    parser.add_option('-n', '', metavar='INDEX_FILE',
                      help='Index file (Gromacs style) for specifying '
                      'atom types. If none is given, all atoms will be '
                      'considered identical.')
    parser.add_option('','--tc', metavar='CORR_TIME', type='float',
                      help='Correlation time (ps) to consider.')
    parser.add_option('','--nc', metavar='CORR_STEPS', type='int',
                      help='Number of time correlation steps to consider.')
    parser.add_option('','--Nq', metavar='QPOINTS', type='int',
                      default=20000,
                      help='Approximate maximum number of q points sampled. '
                      'QPOINTS=-1 implies no limit.')
    parser.add_option('','--q-max', metavar='KMAX', type='float', default=50,
                      help='Largest inverse length to consider (in nm^-1)')
    parser.add_option('','--max-frames', metavar='NFRAMES', type='int',
                      default=0,
                      help='Read no more than NFRAMES frames from trajectory file')
    parser.add_option('','--step', metavar='NSTEP', type='int', default=1,
                      help='Only use every (NSTEP)th trajectory frame. '
                      'Default NSTEP is 1.')
    parser.add_option('','--stride', metavar='NSTRIDE', type='int', default=1,
                      help='Stride NSTRIDE frames inbetween nc.')
    parser.add_option('-v', '--verbose', action='store_true', 
                      default=False,
                      help='Verbose output')

    (options, args) = parser.parse_args()
    if not options.f:
        parser.print_help()
        sys.exit()

    if options.tc and options.nc:
        print('Options --tc and --nc are mutuly exclusive')
        sys.exit(1)

    if options.f is None:
        print('A trajectory must be specified with option -f')
        sys.exit(1)

    if options.f.endswith('.xtc'):
        trajectory_reader = XTC_reader
    elif re.match(r'^.+\.trj(\.(gz|bz2))?$',options.f):
        trajectory_reader = curry(TRJ_reader,
                                  x_factor=0.1, t_factor=0.001)
    else:
        print('Unknown trajectory format')
        sys.exit(1)

    traj = trajectory_reader(options.f)
    f0 = traj.next()
    reference_box = f0['box']
    f1 = traj.next()
    delta_t = f1['time'] - f0['time']
    if options.verbose: 
        print('delta_t found to be %f [ps] --> f_max = %f [GHz]' % \
                  (delta_t, 1000.0/delta_t))
    traj.close()

    
    if options.tc:
        assert(options.tc >= 0.0)
        N_tc = 1 + int(ceil(options.tc/delta_t))
        N_tc += N_tc%2
    elif options.nc:
        assert(options.nc >= 1)
        N_tc = options.nc
    else:
        N_tc = 1

    if options.verbose:
        if N_tc > 1:
            tc = delta_t*(N_tc-1)
            print('tc is %f [ps] --> f_min = %f [GHz]' % \
                      (tc, 1000.0/tc))
            print('0..f_max covered by %i points' % N_tc)

    if options.verbose:
        print('Simulation box = \n%s' % str(reference_box))

    rec = reciprocal(reference_box, N_max=options.Nq, q_max=options.q_max)
                     
    if options.verbose:
        print('Nq points = %i' % len(rec.kdist))
        print('q_max = %f --> x_min = %f' % (options.q_max, 1.0/options.q_max))


    def add_rho_ks(frame):
        frame['rho_ks'] = [calc_rho_k(x, rec.kvals) for x in frame['xs']]
        return frame

    
    assert options.stride > 0
    N_s = options.stride

    traj = trajectory_reader(options.f, index_file=options.n)
    itraj = traj
    if options.step > 1:
        pass
    if options.max_frames > 0:
        itraj = islice(itraj, options.max_frames)

    try:
        frame_list = deque(islice(itraj, N_tc), N_tc)
    except StopIteration:
        print('Failed to read %i frames (minimum required) from %s' % \
                  (N_tc, options.f))
        sys.exit(1)

    for i, f in enumerate(frame_list):
        frame_list[i] = add_rho_ks(f)

    # * Assert box is not changed during consecutive frames
    # * Handle different time steps?

    Ntypes = len(traj.types)
    m = count(0)
    mij_list = [(m.next(),i,j) for i in range(Ntypes) for j in range(i,Ntypes)]
    type_combos = [traj.types[i]+'-'+traj.types[j] for _, i, j in mij_list]

    F_q_t_avs = [averager(np.zeros(len(rec.kdist)), N_tc) for _ in mij_list]

    while len(frame_list) > 0:
        rho_q_0 = frame_list[0]['rho_ks']
        for time_i, frame in enumerate(frame_list):
            rho_q_i = frame['rho_ks']
            for m, i, j in mij_list:
                F_q_t_avs[m].add(np.real(rho_q_0[i]*rho_q_i[j].conjugate()), time_i)
        
        # Explicitly pop frames to ensure proper stride to next frame_list[0] 
        npopleft(frame_list, N_s)
        
        # Skip a few trajectory frames if N_s > N_tc
        consume(itraj, max((0, N_s-N_tc)))

        for f in islice(itraj, min((N_tc, N_s))):
            if options.verbose: 
                sys.stdout.write("Adding time: %f\r" % f['time'])
                sys.stdout.flush()
            frame_list.append(add_rho_ks(f))


                
    F_q_t_full = [np.array([F_q_t_avs[m].get_av(time_i) for time_i in range(N_tc)])
                  for m, _, _ in mij_list]

    # dirty smooth it out
    pts = 100
    delta_k = rec.kdist[1]
    max_k = options.q_max
    rng = (-delta_k/4, max_k+delta_k/4) 
    Npoints, edges = np.histogram(rec.kdist, bins=pts, range=rng)
    F_q_t = [np.zeros((N_tc, pts)) for _ in mij_list]
    F_q_t_sd = [np.zeros((N_tc, pts)) for _ in mij_list]
    k = 0.5*(edges[1:]+edges[:-1])
    t = delta_t*np.arange(N_tc)
    
    for m,_,_ in mij_list:
        ci = 0
        for i, n in enumerate(Npoints):
            if n == 0:
                F_q_t[m][:,i] = np.NaN
                continue
            s = F_q_t_full[m][:,ci:ci+n]
            F_q_t[m][:,i] = np.mean(s, axis=1)
            F_q_t_sd[m][:,i] = np.std(s, axis=1)
            ci += n
