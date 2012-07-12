
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

__all__ = ['create_mfile', 'create_pfile']

import numpy as np
import logging
from itertools import repeat

logger = logging.getLogger('dynsf')

def create_mfile(filename, output, comment=None):
    with open(filename, 'w') as fh:
        popts = np.get_printoptions()
        np.set_printoptions(threshold='inf',
                            linewidth='inf')

        if comment is not None:
            fh.write('%%%\n')
            fh.write(''.join(['% '+x+'\n' for x in comment.split('\n')]))
            fh.write('%%%\n')

        for v, n, desc in output:
            fh.write("\n%% %s\n%s = ...\n%s;\n" % (desc, n, str(v)))

        np.set_printoptions(**popts)
        logger.info('Wrote Matlab-style output to %s' % fh.name)

def create_pfile(filename, output, comment=None):
    import cPickle
    with open(filename, 'w') as fh:
        cPickle.dump(output, fh)
        logger.info('Wrote pickled output to %s' % fh.name)
