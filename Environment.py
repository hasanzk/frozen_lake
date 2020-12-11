

import numpy as np
from EnvironmentModel import EnvironmentModel

class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        print("Environment initialized...")
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)

    # resets the state of the environment and the number of steps taken
    def reset(self):
        self.n_steps = 0
        self.state = 0 

        return self.state

    # receives an action and returns a state drawn according to p together with the corresponding..
    # expected reward and whether the max steps has been exceeded or not
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    # renders the environment
    def render(self, policy=None, value=None):
        raise NotImplementedError()