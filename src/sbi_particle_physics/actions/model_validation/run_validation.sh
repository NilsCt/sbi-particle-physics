#!/bin/sh

echo "Running on $(hostname)"
echo "Arguments received: $@"

source ~/.bashrc
conda activate mlhep

python -m sbi_particle_physics.actions.model_validation.model_validation "$@"