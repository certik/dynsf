#!/usr/bin/env python

import re
import sys
import numpy as np

from itertools import count, islice, imap
from collections import deque

from traj_io import get_itraj, iwindow
from filon import filonc
from reciprocal import reciprocal


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

    iframe = get_itraj(options.f, step=options.step)
    
    f0, f1 = islice(iframe, 2) 
    delta_t = f1['time'] - f0['time']
    reference_box = f0['box']
    types = f0['types']

    if 'vs' in f0:
        calculate_current = True
    else:
        calculate_current = False

    if options.verbose: 
        print('delta_t found to be %f [fs] --> f_max = %f [THz]' % \
                  (delta_t, 1.0/delta_t))

    
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
            print('tc is %f [ps] --> f_min = %f [THz]' % \
                      (tc, 1.0/tc))
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

    itraj = get_itraj(options.f, index_file=options.n, 
                            step=options.step, max_frames=options.max_frames)

    traj_window = iwindow(itraj, width=N_tc, stride=options.stride,
                          map_item=rec.process_frame)


    # TODO....
    # * Assert box is not changed during consecutive frames
    # * Handle different time steps?


    m = count(0)
    mij_list = [(m.next(),i,j) 
                for i in range(len(types)) for j in range(i,len(types))]
    type_pairs = [types[i]+'-'+types[j] for _, i, j in mij_list]

    F_q_t_avs = [averager(np.zeros(len(rec.qdist)), N_tc) for _ in mij_list]
    if calculate_current:
        Cl_q_t_avs = [averager(np.zeros(len(rec.qdist)), N_tc) for _ in mij_list]
        Ct_q_t_avs = [averager(np.zeros(len(rec.qdist)), N_tc) for _ in mij_list]

    for frame_list in traj_window:
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
    #F_q_t_sd = [np.zeros((N_tc, pts)) for _ in mij_list]
    if calculate_current:
        Cl_q_t =    [np.zeros((N_tc, pts)) for _ in mij_list]
        #Cl_q_t_sd = [np.zeros((N_tc, pts)) for _ in mij_list]
        Ct_q_t =    [np.zeros((N_tc, pts)) for _ in mij_list]
        #Ct_q_t_sd = [np.zeros((N_tc, pts)) for _ in mij_list]
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
            else:
                s = F_q_t_full[m][:,ci:ci+n]
                F_q_t[m][:,i] = np.mean(s, axis=1)
                #F_q_t_sd[m][:,i] = np.std(s, axis=1)
                if calculate_current:
                    s = Cl_q_t_full[m][:,ci:ci+n]
                    Cl_q_t[m][:,i] = np.mean(s, axis=1)
                    #Cl_q_t_sd[m][:,i] = np.std(s, axis=1)
                    s = Ct_q_t_full[m][:,ci:ci+n]
                    Ct_q_t[m][:,i] = np.mean(s, axis=1)
                    #Ct_q_t_sd[m][:,i] = np.std(s, axis=1)

                ci += n
