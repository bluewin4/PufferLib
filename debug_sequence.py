#!/usr/bin/env python3
"""Debug script to test sequence_learn environment"""

import numpy as np

try:
    from pufferlib.ocean.sequence_learn import SequenceLearn
    print("✓ Import successful")
    
    # Test environment creation
    env = SequenceLearn(num_envs=1, reward_rule=3, production_rule=0)
    print("✓ Environment created")
    print(f"  Observation space: {env.single_observation_space}")
    print(f"  Action space: {env.single_action_space}")
    
    # Test reset
    obs, _ = env.reset()
    print("✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation: {obs}")
    
    # Test a few steps
    for i in range(5):
        action = np.array([np.random.randint(0, 2)])
        obs, rewards, terminals, truncations, info = env.step(action)
        print(f"  Step {i}: action={action[0]}, reward={rewards[0]:.3f}, obs_shape={obs.shape}")
        
        if terminals[0]:
            print("  Episode terminated")
            break
    
    env.close()
    print("✓ All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 