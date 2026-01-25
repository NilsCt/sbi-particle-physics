#!/bin/sh
echo "Running on $(hostname)"
echo "Arguments received (raw): $@"

PROCESS="$1"
shift

BASE_INDEX=10000
AMOUNT_PER_WORKER=1
START_INDEX=$((BASE_INDEX + PROCESS * AMOUNT_PER_WORKER))

echo "PROCESS=${PROCESS}"
echo "Computed START_INDEX=${START_INDEX}"

source ~/.bashrc
conda activate mlhep

echo "About to execute python command:"
python -m sbi_particle_physics.actions.data_generation.data_generation \
  --start-index "${START_INDEX}" \
  --amount "${AMOUNT_PER_WORKER}" \
  "$@"
