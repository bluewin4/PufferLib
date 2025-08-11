#!/bin/bash
# Script to configure and rebuild sequence_learn with different parameters

SEQUENCE_LENGTH=${1:-2}
NUM_VALUES=${2:-2}
TRIALS_PER_EPISODE=${3:-32}

echo "Configuring sequence_learn with:"
echo "  SEQUENCE_LENGTH: $SEQUENCE_LENGTH"
echo "  NUM_VALUES: $NUM_VALUES" 
echo "  TRIALS_PER_EPISODE: $TRIALS_PER_EPISODE"

# Backup original files
cp pufferlib/ocean/sequence_learn/sequence_learn.h pufferlib/ocean/sequence_learn/sequence_learn.h.bak
cp pufferlib/ocean/sequence_learn/sequence_learn.py pufferlib/ocean/sequence_learn/sequence_learn.py.bak

# Update the header file with new values
sed -i '' "s/#define SEQUENCE_LENGTH [0-9]*/#define SEQUENCE_LENGTH $SEQUENCE_LENGTH/" pufferlib/ocean/sequence_learn/sequence_learn.h
sed -i '' "s/#define NUM_VALUES [0-9]*/#define NUM_VALUES $NUM_VALUES/" pufferlib/ocean/sequence_learn/sequence_learn.h
sed -i '' "s/#define TRIALS_PER_EPISODE [0-9]*/#define TRIALS_PER_EPISODE $TRIALS_PER_EPISODE/" pufferlib/ocean/sequence_learn/sequence_learn.h

# Update the Python wrapper to match
sed -i '' "s/self.sequence_length = [0-9]*/self.sequence_length = $SEQUENCE_LENGTH/" pufferlib/ocean/sequence_learn/sequence_learn.py
sed -i '' "s/self.num_values = [0-9]*/self.num_values = $NUM_VALUES/" pufferlib/ocean/sequence_learn/sequence_learn.py

# Note: OBSERVATION_SIZE is calculated automatically as (SEQUENCE_LENGTH + 3) in both files

# Rebuild
echo "Rebuilding C environment..."
bash scripts/build_ocean.sh sequence_learn

echo ""
echo "Done! Configuration updated for both C and Python."
echo "Run standalone with: ./sequence_learn [reward_rule] [production_rule]"
echo "  reward_rule: 0=exact, 1=partial, 2=exponential, 3=dense (default: 3)"
echo "  production_rule: 0=flat, 1=associative (default: 0)"
echo ""
echo "Train with: ./train_sequence_4x3.sh (or create your own training script)" 