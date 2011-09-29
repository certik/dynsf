#!/usr/bin/env python

from itertools import count, islice
from collections import deque
import re
import sys
import numpy as np

from traj_io import trajectory_iterator
from rho_j_q import calc_rho_q, calc_rho_j_q

def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # From the python.org
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)

def npopleft(deq, n):
    "Left-pop (and discard) n items from deq, or clear deq."
    if len(deq) >= n:
        for _ in range(n):
            deq.popleft()
    else:
        deq.clear()


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
    def __init__(self, box, N_max=-1, q_max=10.0, debug=False):
        """Create a set of reciprocal coordinates, and calculate rho_q/j_q 
        for a trajectory frame.

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

        self.debug = debug

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
        qdist = np.sqrt(np.sum(kvals**2, axis=0))*(1.0/(2*np.pi))
        I = np.nonzero(qdist<=q_max)[0]
        I = I[qdist[I].argsort()]
        qdist = qdist[I[1:]]
        kvals = kvals[:,I[1:]]
        if not self.q_prune is None:
            N = len(qdist)
            p = np.ones(N)
            # N(k) = a k^3
            # N'(k) = 3a k^2
            p = (self.q_prune/qdist)**2
            I = np.nonzero(p > np.random.rand(N))[0]
            qdist = qdist[I]
            kvals = kvals[:,I]
        self.kvals = kvals
        self.qdist = qdist
        N = len(qdist)
        self.kdirect = kvals / (2.0*np.pi*qdist.reshape((1,N)))

    def process_frame(self, frame):
        if self.debug: 
            sys.stdout.write("processing frame at time = %f\r" % f['time'])
            sys.stdout.flush()
        if 'vs' in frame:
            rho_qs, j_qs = zip(*[calc_rho_j_q(x, v, self.kvals) 
                                 for x,v in zip(frame['xs'],frame['vs'])])
            jz_qs = [np.sum(j*self.kdirect, axis=0) for j in j_qs]
            frame['j_qs'] = j_qs
            frame['jz_qs'] = jz_qs
            frame['jpar_qs'] = [j-(jz*self.kdirect) for j,jz in zip(j_qs, jz_qs)]
            frame['rho_qs'] = rho_qs
        else:
            frame['rho_qs'] = [calc_rho_q(x, self.kvals) for x in frame['xs']]
        return frame

        

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
#    parser.add_option('','--tc', metavar='CORR_TIME', type='float',
#                      help='Correlation time (ps) to consider.')
    parser.add_option('','--nc', metavar='CORR_STEPS', type='int',
                      help='Number of time correlation steps to consider.')
    parser.add_option('','--Nq', metavar='QPOINTS', type='int',
                      default=40000,
                      help='Approximate maximum number of q points sampled. '
                      'QPOINTS=-1 implies no limit.')
    parser.add_option('','--q-bins', metavar='BINS', type='int',
                      default=80, help='Number of bins used along the q-axis for '
                      'the final result.')
    parser.add_option('','--q-max', metavar='KMAX', type='float', default=20,
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

#    if options.tc and options.nc:
#        print('Options --tc and --nc are mutuly exclusive')
#        sys.exit(1)

    if options.f is None:
        print('A trajectory must be specified with option -f')
        sys.exit(1)

    iframe = trajectory_iterator(options.f, step=options.step)
    
    f0, f1 = islice(iframe, 2) 
    delta_t = f1['time'] - f0['time']
    reference_box = f0['box']

    if 'vs' in f0:
        calculate_current = True
    else:
        calculate_current = False

    if options.verbose: 
        print('delta_t found to be %f [fs] --> f_max = %f [GHz]' % \
                  (delta_t, 1000.0/delta_t))

    
#    if options.tc:
#        assert(options.tc >= 0.0)
#        N_tc = 1 + int(ceil(options.tc/delta_t))
#        N_tc += N_tc%2
#    elif options.nc:
    if options.nc:
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

    rec = reciprocal(reference_box, N_max=options.Nq, q_max=options.q_max,
                     debug=options.verbose)
                     
    if options.verbose:
        print('Nq points = %i' % len(rec.qdist))
        print('q_max = %f --> x_min = %f' % (options.q_max, 1.0/options.q_max))

    
    assert options.stride > 0
    N_stride = options.stride

    itraj = trajectory_iterator(options.f, index_file=options.n, 
                                step=options.step, max_frames=options.max_frames)

    frame_list = deque(islice(itraj, N_tc), N_tc)
    if len(frame_list) < N_tc:
        raise RuntimeError('Failed to read %i frames (minimum required) from %s' % \
                  (N_tc, options.f))

    for i, f in enumerate(frame_list):
        frame_list[i] = rec.process_frame(f)

    # TODO....
    # * Assert box is not changed during consecutive frames
    # * Handle different time steps?


    m = count(0)
    types = frame_list[0]['types']
    mij_list = [(m.next(),i,j) for i in range(len(types)) for j in range(i,len(types))]
    type_pairs = [types[i]+'-'+types[j] for _, i, j in mij_list]

    F_q_t_avs = [averager(np.zeros(len(rec.qdist)), N_tc) for _ in mij_list]
    if calculate_current:
        Cl_q_t_avs = [averager(np.zeros(len(rec.qdist)), N_tc) for _ in mij_list]
        Ct_q_t_avs = [averager(np.zeros(len(rec.qdist)), N_tc) for _ in mij_list]

    while len(frame_list) > 0:
        if calculate_current:
            rho_q_0s = frame_list[0]['rho_qs']
            jz_q_0s = frame_list[0]['jz_qs']
            jpar_q_0s = frame_list[0]['jpar_qs']
            for time_i, frame in enumerate(frame_list):
                rho_q_is = frame['rho_qs']
                jz_q_is = frame['jz_qs']
                jpar_q_is = frame['jpar_qs']
                for m, i, j in mij_list:
                    F_q_t_avs[m].add(np.real(rho_q_0s[i] * rho_q_is[j].conjugate()), time_i)
                    Cl_q_t_avs[m].add(np.real(jz_q_0s[i] * jz_q_is[j].conjugate()), time_i)
                    Ct_q_t_avs[m].add(0.5 * np.real(np.sum(jpar_q_0s[i] * \
                                                               jpar_q_is[j].conjugate(),
                                                           axis=0)), 
                                      time_i)
        else:
            rho_q_0s = frame_list[0]['rho_qs']
            for time_i, frame in enumerate(frame_list):
                rho_q_is = frame['rho_qs']
                for m, i, j in mij_list:
                    F_q_t_avs[m].add(np.real(rho_q_0s[i] * rho_q_is[j].conjugate()), time_i)
        
        # Explicitly pop frames to ensure proper stride to next frame_list[0] 
        npopleft(frame_list, N_stride)
        
        # Skip N_stride-N_tc frames if N_stride > N_tc
        consume(itraj, max((0, N_stride-N_tc)))

        # Append new frames (if there are any left) to the deque
        for f in islice(itraj, min((N_tc, N_stride))):
            frame_list.append(rec.process_frame(f))


                
    F_q_t_full = [np.array([F_q_t_avs[m].get_av(time_i) for time_i in range(N_tc)])
                  for m, _, _ in mij_list]
    if calculate_current:
        Cl_q_t_full = [np.array([Cl_q_t_avs[m].get_av(time_i) for time_i in range(N_tc)])
                       for m, _, _ in mij_list]
        Ct_q_t_full = [np.array([Ct_q_t_avs[m].get_av(time_i) for time_i in range(N_tc)])
                       for m, _, _ in mij_list]

    # naive smooth it out
    pts = options.q_bins
    delta_q = rec.qdist[1]
    max_q = options.q_max
    rng = (-delta_q/4, max_q+delta_q/4) 
    Npoints, edges = np.histogram(rec.qdist, bins=pts, range=rng)
    F_q_t =    [np.zeros((N_tc, pts)) for _ in mij_list]
    F_q_t_sd = [np.zeros((N_tc, pts)) for _ in mij_list]
    if calculate_current:
        Cl_q_t =    [np.zeros((N_tc, pts)) for _ in mij_list]
        Cl_q_t_sd = [np.zeros((N_tc, pts)) for _ in mij_list]
        Ct_q_t =    [np.zeros((N_tc, pts)) for _ in mij_list]
        Ct_q_t_sd = [np.zeros((N_tc, pts)) for _ in mij_list]
    q = 0.5*(edges[1:]+edges[:-1])
    t = delta_t*np.arange(N_tc)
    
    for m,_,_ in mij_list:
        ci = 0
        for i, n in enumerate(Npoints):
            if n == 0:
                F_q_t[m][:,i] = np.NaN
                if calculate_current:
                    Cl_q_t[m][:,i] = np.NaN
                    Ct_q_t[m][:,i] = np.NaN
                continue
            s = F_q_t_full[m][:,ci:ci+n]
            F_q_t[m][:,i] = np.mean(s, axis=1)
            F_q_t_sd[m][:,i] = np.std(s, axis=1)
            if calculate_current:
                s = Cl_q_t_full[m][:,ci:ci+n]
                Cl_q_t[m][:,i] = np.mean(s, axis=1)
                Cl_q_t_sd[m][:,i] = np.std(s, axis=1)
                s = Ct_q_t_full[m][:,ci:ci+n]
                Ct_q_t[m][:,i] = np.mean(s, axis=1)
                Ct_q_t_sd[m][:,i] = np.std(s, axis=1)

            ci += n
