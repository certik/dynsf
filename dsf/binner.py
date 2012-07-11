
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

__all__ = ['fixed_bin_averager']

import numpy as np
import logging
logger = logging.getLogger('dynsf')


class fixed_bin_averager:
    """Class for averaging data sets of points (x,y) with
       a priori decided positions (x), over a pre-defined number
       of equally sized, linearely distriuted bins.

       This is used to get an average of y for a set of
       lineary spaced x values when a possibly large set
       of {y(x)} is given.
    """
    def __init__(self, x_max, x_bins, x_distances, x_min=0.0):
        assert x_max > x_min
        assert x_bins > 1

        self.delta_x = (x_max-x_min) / (x_bins-1)
        x_range = (x_min-self.delta_x/2, x_max+self.delta_x/2)
        bin_count, edges = np.histogram(x_distances,
                                        bins=x_bins,
                                        range=x_range)
        self.x_linspace = 0.5 * (edges[1:]+edges[:-1])

        I = np.nonzero(bin_count)
        self.bin_count = bin_count[I]
        self.x = self.x_linspace[I]
        self.input_length = len(x_distances)
        self.bins = len(self.x)
        if self.bins != x_bins:
            logger.info('Ignoring %d bins without coverage' % (
                    x_bins-self.bins))

    def bin(self, y, axis=0):
        y = np.require(y)
        assert y.shape[axis] == self.input_length

        res_shape = list(y.shape)
        res_shape[axis] = self.bins
        result = np.zeros(res_shape)

        ci = 0
        ind_x = [slice(None)]*len(y.shape)
        ind_y = [slice(None)]*len(y.shape)
        for i, n in enumerate(self.bin_count):
            ind_x[axis] = i
            ind_y[axis] = slice(ci, ci+n)
            result[ind_x] = np.mean(y[ind_y], axis=axis)
            ci += n

        return result
