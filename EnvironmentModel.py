

import numpy as np

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        print("EnvironmentModel initialized...")
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    # returns the probability of transitioning from state to next state given action
    def p(self, next_state, state, action):
        raise NotImplementedError()

    # returns the expected reward in having transitioned from state to next state given action
    def r(self, next_state, state, action):
        raise NotImplementedError()

    # receives a pair of state and action and returns a state drawn according to p together with the corresponding expected reward
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        
        # However, with probability 0.1, the environment ignores the desired direction and the agent slips (moves one tile in a random direction)
        uncertanity = 0.1/15
        for idx,c in enumerate(p):
            if c == 1:
                p[idx] = c - 0.1
            else:
                p[idx] = uncertanity
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward

