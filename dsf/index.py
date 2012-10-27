
# Copyright (C) 2012 Mattias Slabanja <slabanja@chalmers.se>
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

import re
import numpy as np
from os.path import isfile

section_re = re.compile(r'^ *\[ *([a-zA-Z0-9_.-]+) *\] *$')

class section_index:
    """Read an ini-style gromacs index file

    Reads and parses named index file, keep a list
    of name-array-tuples, containing
    name and indices of the specified (non-empty) sections.
    """
    def __init__(self, filename, max_index):

        if filename is not None and isfile(filename):
            sections = []
            members = []
            name = None
            with open(filename, 'r') as f:
                for L in f:
                    m = section_re.match(L)
                    if m:
                        if members and name:
                            sections.append((name, np.unique(np.concatenate(members))))
                        name = m.group(1)
                        members = []
                    elif not L.isspace():
                        members.append(np.fromstring(L, dtype=int, sep=' '))
                if members and name:
                    sections.append((name, np.unique(np.concatenate(members))-1))
        else:
            sections = [("all", np.arange(max_index, dtype=int))]

        self.sections = sections
        if not self.valid_index_limits(max_index):
            raise ValueError("section_index: Index file seems to contain one or more invalid indices. "
                              "For the provided trajectory file, indices must be in range [1, %d]" % (max_index,))

    def valid_index_limits(self, N):
        for _,I in self.sections:
            if I[0] < 0 or I[-1] >= N:
                return False
        return True

    def get_section_names(self):
        return [n for n,_ in self.sections]

    def get_section_indices(self):
        return [i for _,i in self.sections]

    def N_sections(self):
        return len(self.sections)

    def get_section_split_function(self):
        """Special function for splitting (3,N) dimensioned x or v arrays

        Split x/v into list of xs/vs in accordance with specified sections.
        """
        indices = [I for _,I in self.sections]
        def fun(frame):
            frame = frame.copy()
            frame['xs'] = [frame['x'][:,I] for I in indices]
            if 'v' in frame:
                frame['vs'] = [frame['v'][:,I] for I in indices]
            return frame
        return fun
