
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

__all__ = ['get_itraj', 'iwindow']

from numpy import pi, sin, cos, arange, array, zeros
import numpy as np
import sys
import logging
from itertools import islice, imap, count
from os.path import isfile
from collections import deque

from trajectory_readers import *

logger = logging.getLogger('dynsf')

def get_itraj(filename, step=1, max_frames=0,
              readers=trajectory_readers):
    """Return a dynsf-style trajectory iterator

    Simple wrapper for the trajectory_reader-classes.

    step: (1 by default = every single frame), must be > 0.

    max_frames: (0 by default = no limit), must be >= 0.

    Each iterator step consists of a dictionary.
    {
     'index' : trajectory frame index (1, 2, 3, ...),
     'box'   : simulation box as 3 row vectors (nm),
     'N'     : number of atoms,
     'x'     : particle positions as 3xN array (nm),
     'v'     : (*) particle velocities as 3xN array (nm/ps),
     'time'  : (*) simulation time (ps),
    }
    (*) may not be available, depends on reader and trajectory file format.
    """

    assert step > 0
    assert max_frames >= 0
    if max_frames == 0:
        max_frames = sys.maxint
    elif step > 1:
        max_frames = max_frames*step

    if not isfile(filename):
        raise IOError('File "%s" does not exist' % filename)

    for reader in readers:
        # Simply pick the first reader that seems to work
        if reader.reader_available():
            reader_name = reader.__name__
            try:
                logger.debug('Trying trajectory_reader %s' % reader_name)
                i = reader(filename)
                return islice(reader(filename), 0, max_frames, step)
            except Exception as e:
                logger.debug('Trying trajectory_reader %s failed to open file %s' % (
                        reader_name, filename))

    raise IOError("Failed to open trajectory file %s" % filename)


def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # From the python.org
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)


class iwindow:
    """Sliding window iterator

    Returns consecutive windows (a windows is represented as a list
    of objects), created from an input iterator.

    Variable width (length of window, default 2),
    and stride (distance between the start of two consecutive
    window frames, default 1).
    Optional element_processor to process each non-discarded object.
    Useful if stride > width and map_item is expensive (as compared to
    directly passing imap(fun, itraj) as itraj).
    If stride < width, you could as well directly pass "imap(fun, itraj)"
    """
    def __init__(self, itraj, width=2, stride=1, element_processor=None):

        self._raw_it = itraj
        if element_processor:
            self._it = imap(element_processor, self._raw_it)
        else:
            self._it = self._raw_it
        assert(stride >= 1)
        assert(width >= 1)
        self.width = width
        self.stride = stride
        self._window = None

    def __iter__(self):
        return self

    def next(self):
        if self._window is None:
            self._window = deque(islice(self._it, self.width), self.width)
        else:
            if self.stride >= self.width:
                self._window.clear()
                consume(self._raw_it, self.stride-self.width)
            else:
                for _ in xrange(min((self.stride, len(self._window)))):
                    self._window.popleft()
            for f in islice(self._it, min((self.stride, self.width))):
                self._window.append(f)

        if len(self._window) == 0:
            raise StopIteration

        return list(self._window)

