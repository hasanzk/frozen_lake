

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

    def reset(self):
        self.n_steps = 0
        self.state = 0 

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()




def argmax_random(A):
    arg = np.argsort(A)[::-1]
    n_tied = sum(np.isclose(A, A[arg[0]]))
    return np.random.choice(arg[0:n_tied])

class EGreedySelection:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, q, actions):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(actions)
        else:
            return actions[argmax_random(q[actions])]
