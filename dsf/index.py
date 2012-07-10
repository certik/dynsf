
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
import np

section_re = re.compile(r'^ *\[ *([a-zA-Z0-9_.-]+) *\] *$')

class section_index:
    """Read an ini-style gromacs index file

    Reads and parses named index file, keep a list
    of name-array-tuples, containing
    name and indices of the specified (non-empty) sections.
    """
    def __init__(self, filename=None):
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
            sections.append((name, np.unique(np.concatenate(members))))

        self.sections = sections

    def valid_index_limits(self, N):
        for _,I in self.sections:
            if I[0] < 0 or I[-1] >= N:
                return False
        return True

    def get_section_names(self):
        if self.sections == []:
            return ["all"]
        else:
            return [n for n,_ in self.sections]

    def get_section_split_function(self):
        """Special function for splitting (3,N) dimensioned x or v arrays

        Split x/v into list of xs/vs in accordance with specified sections.
        """
        if self.sections == []:
            def split(frame):
                frame['xs'] = [frame['x']]
                if 'v' in frame:
                    frame['vs'] = [frame['v']]
                return frame
        else:
            indices = [I for _,I in self.sections]
            def split(frame):
                frame['xs'] = [frame['x'][:,I] for I in indices]
                if 'v' in frame:
                    frame['vs'] = [frame['v'][:,I] for I indices]
                return frame
        return split
