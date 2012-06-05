
extra_compile_args = ['-fopenmp', '-O3', '-std=c99']
extra_link_args = ['-fopenmp']

# Let local_compiler be None in order to use the default compiler
#
local_compiler = None
local_linker = local_compiler


# Example: Explicitly use gcc
#
local_compiler = 'gcc'
extra_compile_args = ['-fPIC', '-fopenmp', '-Ofast', '-march=native', '-std=c99']

local_linker = local_compiler
local_link_shared = ['-shared']
extra_link_args = ['-fopenmp']


# # Example: Use icc instead of the default compiler
# #
# local_compiler = 'icc'
# extra_compile_args = ['-openmp', '-xHOST', '-O3', '-fno-alias', '-fPIC']

# local_linker = local_compiler
# local_link_shared = ['-shared']
# extra_link_args = ['-openmp']


# # Example: Use pgcc instead of the default compiler
# #
# local_compiler = 'pgcc'
# extra_compile_args = ['-mp=numa', '-O4', '-Msafeptr', '-fPIC']

# local_linker = local_compiler
# local_link_shared = ['-shared']
# extra_link_args = ['-mp']
