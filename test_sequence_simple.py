#!/usr/bin/env python3
"""Simple test of the sequence_learn environment without vectorization"""

import numpy as np
from pufferlib.ocean.sequence_learn import SequenceLearn

# Test single environment
print("Testing single SequenceLearn environment...")

# Create environment with partial match reward (easier)
env = SequenceLearn(num_envs=1, reward_rule=1, production_rule=0)
obs, _ = env.reset()

print(f"Single env observation shape: {obs.shape}")
print(f"Expected shape: (1, 6)")
print(f"Initial observation: {obs}")

# Run a few steps
for i in range(20):
    # Random action
    actions = np.array([np.random.randint(0, 4)])
    obs, rewards, terminals, truncations, info = env.step(actions)
    
    if rewards[0] > 0:
        print(f"Step {i}: Got reward {rewards[0]}")
    
    if terminals[0]:
        print(f"Episode completed at step {i}")
        break

env.close()

# Now test with multiple environments
print("\nTesting with 4 environments...")
env = SequenceLearn(num_envs=4, reward_rule=1, production_rule=0)
obs, _ = env.reset()
print(f"Multi env observation shape: {obs.shape}")
print(f"Expected shape: (4, 6)")

env.close()
print("Tests completed!") 