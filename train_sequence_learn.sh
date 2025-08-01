#!/bin/bash
# Training script for sequence learning environment

# Set CUDA device if needed
export CUDA_VISIBLE_DEVICES=0

# Run training with CUDA
python -m pufferlib.pufferl train puffer_sequence_learn \
    --train.device cuda \
    --train.total_timesteps 10000000 \
    --train.learning_rate 0.0003 \
    --train.num_minibatches 8 \
    --train.update_epochs 4 \
    --train.anneal_lr true \
    --train.gamma 0.99 \
    --train.gae_lambda 0.95 \
    --train.ent_coef 0.01 \
    --train.vf_coef 0.5 \
    --train.max_grad_norm 0.5 \
    --train.track true \
    --train.wandb_project sequence_learn \
    --train.wandb_name "sequence_learn_lstm_${HOSTNAME}_$(date +%Y%m%d_%H%M%S)" \
    --train.save_interval 500 \
    --train.checkpoint_interval 5000 \
    --lstm.enabled true \
    --lstm.hidden_size 128 \
    --lstm.num_layers 2 