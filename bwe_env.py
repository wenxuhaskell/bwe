import gymnasium as gym
import numpy as np

# actions space consists of 4 discrete actions
actions = [1, 2, 3, 4]
# state vector (6-dimension)
state = {
    'f1': 0,
    'f2': 0,
    'f3': 0,
    'f4': 0,
    'f5': 0,
    'f6': 0
}

observations = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']


bw_action_space = gym.spaces.Discrete(len(actions))

def make_bw_obs_space():
    lower_obs_bound = {
        'f1': 0,
        'f2': 0,
        'f3': 0,
        'f4': 0,
        'f5': 0,
        'f6': 0
    }
    higher_obs_bound = {
        'f1': np.inf,
        'f2': np.inf,
        'f3': np.inf,
        'f4': np.inf,
        'f5': np.inf,
        'f6': np.inf
    }

    low = np.array([lower_obs_bound[o] for o in observations])
    high = np.array([higher_obs_bound[o] for o in observations])
    shape = (len(observations),)
    return gym.spaces.Box(low,high,shape)

class BandwidthEstimation(gym.Env):
    def __init__(self, max_steps=1000):
        self.actions = actions
        self.observations = observations
        self.action_space = bw_action_space
        self.observation_space = make_bw_obs_space()
        self.seed=0
        self.log = ''
        self.max_steps = max_steps

    def observation(self):
        return np.array([self.state[o] for o in self.observations])

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = {
            'f1': 0,
            'f2': 0,
            'f3': 0,
            'f4': 0,
            'f5': 0,
            'f6': 0
        }
        self.steps_left = self.max_steps

        return self.observation()

    def step(self, action):
        if self.state['time_elapsed'] == 0:
            old_score = 0
        else:
            old_score = self.state['work_done'] / self.state['time_elapsed']

        # Do selected action
        self.actions[action](self.state)
        self.log += f'Chosen action: {self.actions[action].__name__}\n'

        # Do work (TODO)
        #work(self.state)

        new_score = self.state['work_done'] / self.state['time_elapsed']

        reward = new_score - old_score

        self.log += str(self.state) + '\n'

        self.steps_left -= 1
        done = (self.steps_left <= 0)

        return self.observation(), reward, done, {}

    def close(self):
        pass

    def render(self, mode=None):
        print(self.log)
        self.log = ''