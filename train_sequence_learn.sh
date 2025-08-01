#!/bin/bash
# Training script for sequence learning environment

# Set CUDA device if needed
export CUDA_VISIBLE_DEVICES=0

# Run training with CUDA
python -m pufferlib.pufferl train puffer_sequence_learn \
    --train.device cuda \
    --train.total-timesteps 10000000 \
    --train.learning-rate 0.0003 \
    --train.num-minibatches 8 \
    --train.update-epochs 4 \
    --train.anneal-lr true \
    --train.gamma 0.99 \
    --train.gae-lambda 0.95 \
    --train.ent-coef 0.01 \
    --train.vf-coef 0.5 \
    --train.max-grad-norm 0.5 \
    --train.track true \
    --train.wandb-project sequence_learn \
    --train.wandb-name "sequence_learn_lstm_${HOSTNAME}_$(date +%Y%m%d_%H%M%S)" \
    --train.save-interval 500 \
    --train.checkpoint-interval 5000 \
    --lstm.enabled true \
    --lstm.hidden-size 128 \
    --lstm.num-layers 2 