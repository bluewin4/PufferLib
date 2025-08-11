#!/bin/bash
# CPU-optimized training script for 3x3 sequence learning with exponential rewards

echo "Training 3x3 Sequence Learning with Exponential Rewards on CPU"
echo "================================================"

# Run training optimized for CPU
python -m pufferlib.pufferl train puffer_sequence_learn \
    --train.device cpu \
    --train.total-timesteps 50000000 \
    --train.learning-rate 0.001 \
    --train.batch-size 16384 \
    --train.minibatch-size 2048 \
    --train.num-minibatches 8 \
    --train.update-epochs 4 \
    --train.num-steps 256 \
    --train.gamma 0.99 \
    --train.gae-lambda 0.95 \
    --train.ent-coef 0.005 \
    --train.vf-coef 0.5 \
    --train.max-grad-norm 0.5 \
    --train.clip-coef 0.2 \
    --train.anneal-lr true \
    --train.track false \
    --train.verbose true \
    --train.save-interval 1000 \
    --train.checkpoint-interval 10000 \
    --vec.num-envs 256 \
    --vec.num-workers 4 \
    --vec.backend Multiprocessing \
    --lstm.enabled true \
    --lstm.hidden-size 256 \
    --lstm.num-layers 2 \
    --env.reward-rule 2 \
    --env.production-rule 0 