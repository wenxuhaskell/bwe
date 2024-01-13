import gymnasium as gym
import numpy as np
import BweReward

def make_bw_act_space(actions):
    low = 0
    high = max(actions)
    # high = np.inf

    shape = (len(actions),)
    # print(f"action shape {shape} \n")
    bw_action_space = gym.spaces.Box(low, high, shape, dtype=np.float32)
    return bw_action_space


def make_bw_obs_space(observations):
    low = 0.0
    high = np.inf
    shape = (len(observations), len(observations[0]))
    # print(f"state space shape {shape} \n")
    return gym.spaces.Box(low, high, shape, dtype=np.float32)


class BweEnv(gym.Env):
    def __init__(self, dataset, max_steps=10000):
        self._actions = dataset['actions']
        self._observations = dataset['observations']
        self._observations_next = dataset['next_observations']
        self._action_space = make_bw_act_space(self._actions)
        self._rewards = dataset['rewards']
        self._observation_space = make_bw_obs_space(self._observations)
        self._terminals = dataset['terminals']
        self._seed = 0
        self._log = ''
        self._max_steps = len(self._actions)
        self._cur_step = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._cur_step = 0

        return self._observations[self._cur_step], {}

    def step(self, action):
        # Do selected action
        cur_estimate = self._actions[self._cur_step]
        cur_observation = self._observations[self._cur_step]
        next_observation = self._observations_next[self._cur_step]
        cur_reward = self._rewards[self._cur_step]
        cur_terminal = self._terminals[self._cur_step]
        new_reward = BweReward.RewardFunction('QOE_V1')(next_observation)

        self._steps_left -= 1
        if (self._cur_steps >= self._max_steps) or (cur_terminal != 0):
            done = True
        else:
            done = False

        self._log += f'Step: {self._cur_step}\n'
        self._log += f'current estimate {cur_estimate}\n'
        self._log += f'new estimate {action}\n'
        self._log += f'current reward {cur_reward}\n'
        self._log += f'new reward {new_reward}\n'

        truncated = False
        # step count increases
        self._cur_step += 1

        return next_observation, new_reward, done, truncated, {}

    def close(self):
        pass

    def render(self, mode=None):
        print(self._log)
        self._log = ''