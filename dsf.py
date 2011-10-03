#!/usr/bin/env python

import re
import sys
import numpy as np

from itertools import count, islice

from traj_io import get_itraj, iwindow
from filon import filonc
from reciprocal import reciprocal


class averager:
    def __init__(self, N_slots, initial=np.zeros(1)):
        assert(N_slots >= 1)
        self._N = N_slots
        self._data = [np.array(initial) for n in range(N_slots)]
        self._samples = np.zeros(N_slots)
    def __getitem__(self, key):
        return self._data[key]
    def __setitem__(self, key, val):
        self._data[key] = val
        self._samples[key] += 1
    def add(self, array, slot):
        self[slot] += array
    def get_single_av(self, slot):
        f = 1.0/self._samples[slot]
        return f*self._data[slot]
    def get_av(self):
        return np.array([self.get_single_av(i) for i in range(self._N)])
        

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
    parser.add_option('','--nc', metavar='CORR_STEPS', type='int',
                      help='Number of time correlation steps to consider.')
    parser.add_option('','--Nk', metavar='KPOINTS', type='int',
                      default=40000,
                      help='Approximate maximum number of k points sampled. '
                      'KPOINTS=-1 implies no limit.')
    parser.add_option('','--k-bins', metavar='BINS', type='int',
                      default=80, help='Number of bins used along the k-axis for '
                      'the final result.')
    parser.add_option('','--k-max', metavar='KMAX', type='float', default=20,
                      help='Largest inverse length to consider (in 2*pi*nm^-1)')
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

    rec = reciprocal(reference_box, N_max=options.Nk, k_max=options.k_max,
                     debug=options.verbose)
                     
    if options.verbose:
        print('Nk points = %i' % len(rec.qdist))
        print('k_max = %f --> x_min = %f' % (options.k_max, 1.0/options.k_max))

    
    assert options.stride > 0
    N_stride = options.stride

    itraj = get_itraj(options.f, index_file=options.n, 
                            step=options.step, max_frames=options.max_frames)

    itraj_window = iwindow(itraj, width=N_tc, stride=options.stride,
                           map_item=rec.process_frame)


    # TODO....
    # * Assert box is not changed during consecutive frames


    m = count(0)
    mij_list = [(m.next(),i,j) 
                for i in range(len(types)) for j in range(i,len(types))]
    type_pairs = [types[i]+'-'+types[j] for _, i, j in mij_list]

    z = np.zeros(len(rec.qdist))
    F_k_t_avs = [averager(N_tc, z) for _ in mij_list]
    if calculate_current:
        Cl_k_t_avs = [averager(N_tc, z) for _ in mij_list]
        Ct_k_t_avs = [averager(N_tc, z) for _ in mij_list]

    for wind in itraj_window:
        if calculate_current:
            rho_k_0s = wind[0]['rho_ks']
            jz_k_0s = wind[0]['jz_ks']
            jpar_k_0s = wind[0]['jpar_ks']
            for time_i, frame in enumerate(wind):
                rho_k_is = frame['rho_ks']
                jz_k_is = frame['jz_ks']
                jpar_k_is = frame['jpar_ks']
                for m, i, j in mij_list:
                    F_k_t_avs[m][time_i] += np.real(rho_k_0s[i] * rho_k_is[j].conjugate())
                    Cl_k_t_avs[m][time_i] += np.real(jz_k_0s[i] * jz_k_is[j].conjugate())
                    Ct_k_t_avs[m][time_i] += 0.5 * \
                        np.real(np.sum(jpar_k_0s[i] * jpar_k_is[j].conjugate(), axis=0))
        else:
            rho_k_0s = wind[0]['rho_ks']
            for time_i, frame in enumerate(wind):
                rho_k_is = frame['rho_ks']
                for m, i, j in mij_list:
                    F_k_t_avs[m][time_i] += np.real(rho_k_0s[i] * rho_k_is[j].conjugate())


                
    F_k_t_full = [F_k_t_avs[m].get_av() for m, _, _ in mij_list]
    if calculate_current:
        Cl_k_t_full = [Cl_k_t_avs[m].get_av() for m, _, _ in mij_list]
        Ct_k_t_full = [Ct_k_t_avs[m].get_av() for m, _, _ in mij_list]

    # simple smoothing using radially distributed bins
    pts = options.k_bins
    delta_k = rec.kdist[1]
    max_k = options.k_max
    rng = (-delta_k/4, max_k+delta_k/4) 
    Npoints, edges = np.histogram(rec.kdist, bins=pts, range=rng)
    F_k_t =    [np.zeros((N_tc, pts)) for _ in mij_list]
    if calculate_current:
        Cl_k_t =    [np.zeros((N_tc, pts)) for _ in mij_list]
        Ct_k_t =    [np.zeros((N_tc, pts)) for _ in mij_list]
    k = 0.5*(edges[1:]+edges[:-1])
    t = delta_t*np.arange(N_tc)
    
    for m,_,_ in mij_list:
        ci = 0
        for i, n in enumerate(Npoints):
            if n == 0:
                F_k_t[m][:,i] = np.NaN
                if calculate_current:
                    Cl_k_t[m][:,i] = np.NaN
                    Ct_k_t[m][:,i] = np.NaN
            else:
                s = F_k_t_full[m][:,ci:ci+n]
                F_k_t[m][:,i] = np.mean(s, axis=1)
                if calculate_current:
                    s = Cl_k_t_full[m][:,ci:ci+n]
                    Cl_k_t[m][:,i] = np.mean(s, axis=1)
                    s = Ct_k_t_full[m][:,ci:ci+n]
                    Ct_k_t[m][:,i] = np.mean(s, axis=1)
                ci += n
