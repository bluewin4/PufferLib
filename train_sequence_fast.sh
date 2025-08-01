#!/bin/bash
# High-performance training script for sequence learning environment

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run training with optimized settings for GPU
python -m pufferlib.pufferl train puffer_sequence_learn \
    --train.device cuda \
    --train.total-timesteps 5000000 \
    --train.learning-rate 0.0003 \
    --train.batch-size 131072 \
    --train.minibatch-size 16384 \
    --train.num-minibatches 8 \
    --train.update-epochs 4 \
    --train.num-steps 128 \
    --train.gamma 0.99 \
    --train.gae-lambda 0.95 \
    --train.ent-coef 0.01 \
    --train.vf-coef 0.5 \
    --train.max-grad-norm 0.5 \
    --train.clip-coef 0.1 \
    --train.anneal-lr true \
    --train.track false \
    --train.verbose true \
    --vec.num-envs 1024 \
    --vec.num-workers 8 \
    --lstm.enabled true \
    --lstm.hidden-size 128 \
    --lstm.num-layers 2 