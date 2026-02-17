#!/bin/bash

echo "Running on $(hostname)"
echo "Arguments received: $@"

source /home/hep/nrc25/miniconda3/etc/profile.d/conda.sh

conda activate mlhep

echo "Python used:"
which python
python --version

echo "g++ used:"
which g++
g++ --version

echo "PATH:"
echo $PATH

python -m sbi_particle_physics.actions.resume_training.resume_training "$@"
