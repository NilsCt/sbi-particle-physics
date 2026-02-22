#!/bin/bash

echo "Running on $(hostname)"
echo "Arguments received: $@"

source /home/hep/nrc25/miniconda3/etc/profile.d/conda.sh

conda activate mlhep

echo "Python used:"
which python
echo "g++ used:"
which g++

python -m sbi_particle_physics.actions.new_snpe_round.new_snpe_round "$@"
