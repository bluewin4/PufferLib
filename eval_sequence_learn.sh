#!/bin/bash
# Evaluation script for sequence learning environment

# Run evaluation with the latest model
python -m pufferlib.pufferl eval puffer_sequence_learn \
    --train.device cuda \
    --vec.num_envs 1 \
    --load-model-path latest \
    --render_mode human 