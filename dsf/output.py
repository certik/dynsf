
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

__all__ = ['create_mfile']

import numpy as np

def create_mfile(filename, output):
    with open(filename, 'w') as f:
        popts = np.get_printoptions()
        np.set_printoptions(threshold='inf',
                            linewidth='inf')

        for v, n, desc in output:
            f.write("\n%% %s\n%s = \\\n%s;\n" % (desc, n, str(v)))
        
        np.set_printoptions(**popts)
