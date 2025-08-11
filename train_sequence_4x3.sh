#!/bin/bash
# Training script for 3x3 sequence learning with exponential rewards

echo "Training 3x3 Sequence Learning with Exponential Rewards"
echo "======================================================"
echo "PufferLib will auto-detect CPU/GPU and optimize accordingly"
echo ""

# Force CPU training since CUDA is not available
python -m pufferlib.pufferl train puffer_sequence_learn \
    --train.device cpu \
    --train.total-timesteps 50000000 \
    --train.learning-rate 0.001 \
    --train.ent-coef 0.005 \
    --train.anneal-lr true \
    --train.track false \
    --train.verbose true \
    --lstm.enabled true \
    --lstm.hidden-size 256 \
    --lstm.num-layers 2 