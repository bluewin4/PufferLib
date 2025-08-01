#!/usr/bin/env python3
"""Quick test of the sequence_learn environment"""

import numpy as np
from pufferlib.ocean.sequence_learn import SequenceLearn

# Test the environment
print("Testing SequenceLearn environment...")

# Create environment with partial match reward (easier)
env = SequenceLearn(num_envs=4, reward_rule=1, production_rule=0)
obs, _ = env.reset()

print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.single_action_space}")
print(f"Observation space: {env.single_observation_space}")

# Run a few steps
total_rewards = 0
for i in range(100):
    # Random actions
    actions = np.random.randint(0, 4, size=4)
    obs, rewards, terminals, truncations, info = env.step(actions)
    
    total_rewards += rewards.sum()
    
    # Print info when episodes complete
    if terminals.any():
        print(f"Step {i}: Episode completed!")
        print(f"  Rewards: {rewards}")
        print(f"  Info: {info}")

print(f"\nTotal rewards collected: {total_rewards}")
env.close()
print("Test completed successfully!") 