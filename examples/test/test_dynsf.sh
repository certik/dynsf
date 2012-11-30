#!/bin/sh

DYNSF=$HOME/opt/dynsf/bin/dynsf
DIR=$(dirname "$(readlink -f -- $0)")
DATADIR="${DIR}/../data"

declare -i index=1

K_MAX=30        # Consider kspace from gamma (0) to 30 inverse nanometer
K_BINS=20       # Collect result using 20 "bins" between 0 and $K_MAX
TIME_WINDOW=10  # Consider time correlations up to 10 trajectory frames
MAX_FRAMES=99   # Read at most 99 frames from trajectory file (then stop)

# gzipped lammps "custom dump", containing particle positions
TRAJECTORY="${DATADIR}/positions.lammpstrj.gz"
# group particles into species (by indices)
INDEX="${DATADIR}/system.ndx"

${DYNSF} -f "$TRAJECTORY" -n "$INDEX" \
    --k-max=$K_MAX --k-bins=$K_BINS \
    --nt=$TIME_WINDOW --max-frames=$MAX_FRAMES \
    --om=output$((index)).m \
    --op=output$((index++)).pickle


# gzipped lammps "custom dump", containing particle positions in scaled coordinates
TRAJECTORY="${DATADIR}/scaled_positions.lammpstrj.gz"

${DYNSF} -f "$TRAJECTORY" -n "$INDEX" \
    --k-max=$K_MAX --k-bins=$K_BINS \
    --nt=$TIME_WINDOW --max-frames=$MAX_FRAMES \
    --om=output$((index)).m \
    --op=output$((index++)).pickle


# gzipped lammps "custom dump", containing particle positions and velocities
TRAJECTORY="${DATADIR}/positions_and_velocities.lammpstrj.gz"

${DYNSF} -f "$TRAJECTORY" -n "$INDEX" \
    --k-max=$K_MAX --k-bins=$K_BINS \
    --nt=$TIME_WINDOW --max-frames=$MAX_FRAMES \
    --om=output$((index)).m \
    --op=output$((index++)).pickle


# gzipped lammps "custom dump", containing particle positions
TRAJECTORY="${DATADIR}/positions.lammpstrj.gz"
# group particles into species (by indices)
INDEX="${DATADIR}/system.ndx"

${DYNSF} -f "$TRAJECTORY" -n "$INDEX" \
    --k-max=$K_MAX --k-bins=$K_BINS \
    --nt=$TIME_WINDOW --max-frames=$MAX_FRAMES \
    --calculate-self \
    --om=output$((index)).m \
    --op=output$((index++)).pickle
