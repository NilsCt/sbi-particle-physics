#!/bin/bash

echo "Running on $(hostname)"
echo "Arguments received (raw): $@"

PROCESS="$1"
shift

BASE_INDEX=6800
AMOUNT_PER_WORKER=1
START_INDEX=$((BASE_INDEX + PROCESS * AMOUNT_PER_WORKER))

echo "PROCESS=${PROCESS}"
echo "Computed START_INDEX=${START_INDEX}"

source /home/hep/nrc25/miniconda3/etc/profile.d/conda.sh

conda activate mlhep

echo "Python used:"
which python
echo "g++ used:"
which g++

echo "About to execute python command:"

python -m sbi_particle_physics.actions.data_generation.data_generation \
  --start-index "${START_INDEX}" \
  --amount "${AMOUNT_PER_WORKER}" \
  "$@"
