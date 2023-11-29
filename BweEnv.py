import gymnasium as gym
import numpy as np

from BweReward import reward_r3net

def make_bw_act_space(actions):
    low = 0
    high = np.max(actions)
    #high = np.inf

    shape = (len(actions),)
    #print(f"action shape {shape} \n")
    bw_action_space = gym.spaces.Box(low, high, shape, dtype=np.float32)
    return bw_action_space

def make_bw_obs_space(observations):
    low = 0.0
    high = np.inf
    shape = (len(observations),len(observations[0]))
    #print(f"state space shape {shape} \n")
    return gym.spaces.Box(low, high, shape, dtype=np.float32)

class BweEnv(gym.Env):
    def __init__(self, observations, actions, max_steps=1000):
        self.actions = actions
        self.observations = observations
        self.action_space = make_bw_act_space(actions)
        self.observation_space = make_bw_obs_space(observations)
        self.seed=0
        self.log = ''
        self.max_steps = len(actions)
        self.cur_step=0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps_left = self.max_steps
        self.cur_step = 0

        return self.observations[self.cur_step], {}

    def step(self, action):
        # Do selected action (TODO)
        cur_estimate = self.actions[self.cur_step]
        cur_observation = self.observations[self.cur_step]
        self.log += f'bwe: {cur_estimate}\n'

        # Calculate the reward (TODO)
        reward = reward_r3net(cur_observation);

        self.log += str(self.cur_step) + ' . '

        self.steps_left -= 1
        done = (self.steps_left <= 0)
        truncated = False
        # step count increases
        self.cur_step += 1

        return cur_observation, reward, done, truncated, {}

    def close(self):
        pass

    def render(self, mode=None):
        print(self.log)
        self.log = ''