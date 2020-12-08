
import numpy as np
import contextlib
from Environment import Environment
from itertools import product


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)



class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        # Up 0, down 1, left 2, right 3
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        Environment.__init__(self,n_states,n_actions,max_steps,None)


        # Indices to states (coordinates), states (coordinates) to indices 
        self.itos = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        self.stoi = {s: i for (i, s) in enumerate(self.itos)}
        
        self.init_possible_actions()

        # Precomputed transition probabilities
        self._p = np.zeros((self.n_states, self.n_states, self.n_actions))
        
        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):
                next_state = (state[0] + action[0], state[1] + action[1])
                # If next_state is not valid, default to current state index
                next_state_index = self.stoi.get(next_state, state_index)

                # However, with probability 0.1, the environment ignores the desired direction and the agent slips (moves one tile in a random direction)
                for a_idx, a in enumerate(self.actions):
                    if a == action:
                        self._p[next_state_index, state_index, a_idx] += 0.9
                    else:
                        self._p[next_state_index, state_index, a_idx] += self.slip / (self.n_actions - 1)
                    
  
    def init_possible_actions(self):
        self._possible_actions = []
        for s in range(self.n_states - 1):
            s_actions = []
            i, j = self.itos[s]

            if self.valid_coord(i + 1, j):
                s_actions.append(1)
            if self.valid_coord(i - 1, j):
                s_actions.append(0)
            if self.valid_coord(i, j + 1):
                s_actions.append(3)
            if self.valid_coord(i, j - 1):
                s_actions.append(2)

            self._possible_actions.append(s_actions)

    def valid_coord(self, i, j):
        valid = i >= 0 and i < self.lake[0].size and j >= 0 and j < self.lake[0].size
        return valid
    

    def reached_goal(self,s):
        # check if state is within range because of the absorbing state
        if s >= len(self.itos):
            return False
        return self.lake[self.itos[s]] == '$'

    def step(self, action):
        gameover = (self.lake[self.itos[self.state]] == '$') or (self.lake[self.itos[self.state]] == '#')
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done or gameover

        return state, reward, done

    # returns the probability of transitioning from state to next state given action
    def p(self, next_state, state, action):
        # TODO:
        return self._p[next_state, state, action] 

    # returns the expected reward in having transitioned from state to next state given action
    def r(self, next_state, state, action):
        # TODO:
        if not self.reached_goal(state) and self.reached_goal(next_state):
            return 1
        else:
            return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', 'ـ', '<', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            policy = policy.reshape(self.lake.shape)
            for i in range(len(self.lake[0])):
                for j in range(len(self.lake[0])):
                    if self.lake[i][j] == '$':
                        print("$",end=" ")
                    elif self.lake[i][j] == '#':
                        print("#",end=" ")
                    elif self.lake[i, j] == '&':
                        print("&",end=" ")
                    else:
                        print(policy[i][j],end=" ")
                print('')

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
