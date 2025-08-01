'''Sequence learning environment for testing LSTM-RNN ability to infer rules'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.sequence_learn import binding

class SequenceLearn(pufferlib.PufferEnv):
    """
    Sequence learning environment where agents must learn a target sequence
    through multiple trials within an episode.
    
    Parameters:
    - num_envs: Number of parallel environments
    - render_mode: Rendering mode (None or 'human')
    - reward_rule: 0=exact match, 1=partial match, 2=exponential
    - production_rule: 0=flat (all available), 1=associative
    - sequence_length: Length of target sequence (default 4)
    - trials_per_episode: Number of trials per episode (default 16)
    - steps_per_episode: Maximum steps per episode (default 64)
    - num_values: Number of possible values in sequence (default 4)
    """
    def __init__(self, num_envs=1, render_mode=None, log_interval=128,
                 reward_rule=0, production_rule=0, buf=None, seed=0):
        # Constants from C header
        self.sequence_length = 4
        self.num_values = 4
        self.observation_size = self.sequence_length + 2
        
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.observation_size,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(self.num_values)
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            reward_rule=int(reward_rule), production_rule=int(production_rule)
        )
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    import time
    
    # Test different configurations
    configs = [
        {"reward_rule": 0, "production_rule": 0},  # Exact match, flat production
        {"reward_rule": 1, "production_rule": 0},  # Partial match, flat production
        {"reward_rule": 2, "production_rule": 0},  # Exponential, flat production
        {"reward_rule": 0, "production_rule": 1},  # Exact match, associative
    ]
    
    for config in configs:
        print(f"\nTesting config: {config}")
        
        N = 256
        env = SequenceLearn(num_envs=N, **config)
        env.reset()
        
        total_steps = 0
        episodes_completed = 0
        max_rewards_achieved = 0
        total_trials_to_max = []
        
        start = time.time()
        while time.time() - start < 5:  # Run for 5 seconds
            # Random actions for testing
            actions = np.random.randint(0, env.num_values, N)
            obs, rewards, terminals, truncations, info = env.step(actions)
            total_steps += 1
            
            # Track statistics
            for i in range(N):
                if terminals[i] or truncations[i]:
                    episodes_completed += 1
                    log_data = info[0]
                    if 'trials_to_max_reward' in log_data and log_data['trials_to_max_reward'] > 0:
                        max_rewards_achieved += 1
                        total_trials_to_max.append(log_data['trials_to_max_reward'])
        
        elapsed = time.time() - start
        print(f'  Steps per second: {int(N * total_steps / elapsed)}')
        print(f'  Episodes completed: {episodes_completed}')
        print(f'  Episodes with max reward: {max_rewards_achieved}')
        if total_trials_to_max:
            print(f'  Average trials to max reward: {np.mean(total_trials_to_max):.2f}')
        
        env.close() 