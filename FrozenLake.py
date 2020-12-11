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

        # initializes the possible actions for all states of the environment
        self.init_possible_actions()

        # Precomputed transition probabilities
        self._p = np.zeros((self.n_states, self.n_states, self.n_actions))
        
        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):
                # assigning absorbing state as the next state for holes and the goal
                if self.lake[state] == '$' or self.lake[state] == '#':
                    next_state_index = self.absorbing_state
                else:
                    # Compute the coordinates of next state
                    next_state = (state[0] + action[0], state[1] + action[1])
                    # get the index of next state. If next state is not valid, default to current state index
                    next_state_index = self.stoi.get(next_state, state_index)

                # filling p with the probabilities corresponding to the current state, next state and the desired action
                # probability 0.9 is given for the desired action
                # probability 0.1 is distributed on all actions including the desired action as a random move
                # this 0.1 represents the environment ignoring the desired direction and the agent slips (moves one tile in a random direction)
                for a_idx, a in enumerate(self.actions):
                    if a == action:
                        self._p[next_state_index, state_index, a_idx] += 0.9
                    self._p[next_state_index, state_index, a_idx] += self.slip / (self.n_actions)

        # Transition probabilities for the absorbing state
        for a_idx, a in enumerate(self.actions):
            self._p[self.absorbing_state, self.absorbing_state, a_idx] += 1

    # initializes the possible actions for all states of the environment
    def init_possible_actions(self):
        self._possible_actions = []
        for s in range(self.lake.size):
            s_actions = []
            i, j = self.itos[s]

            # checking the validity of all actions from state s
            # if valid, add the action to the list of valid actions
            if self.valid_coord(i - 1, j):
                s_actions.append(0)
            if self.valid_coord(i + 1, j):
                s_actions.append(1)
            if self.valid_coord(i, j - 1):
                s_actions.append(2)
            if self.valid_coord(i, j + 1):
                s_actions.append(3)

            self._possible_actions.append(s_actions)

    # checks if an action is valid or not based on the coordinates of next state
    def valid_coord(self, i, j):
        return i >= 0 and i < self.lake[0].size and j >= 0 and j < self.lake[0].size

    # check whether a given state is the goal or not
    def reached_goal(self,s):
        # check if state is within range because of the absorbing state
        if s >= len(self.itos):
            return False
        return self.lake[self.itos[s]] == '$'

    # receives an action and returns a state drawn according to p together with the corresponding expected..
    # reward and either the agent reached the absorbing state or the max steps has been exceeded or none of them
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    # returns the probability of transitioning from state to next state given action
    def p(self, next_state, state, action):
        return self._p[next_state, state, action]

    # returns the expected reward in having transitioned from state to next state given action
    def r(self, next_state, state, action):
        if self.reached_goal(state):
            return 1
        else:
            return 0

    # renders the state of the environment, the given policy and the given value matrix
    def render(self, policy=None, value=None):
        # render the state of the environment if no policy is given
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                # assigning the agent coordinates to agent symbol
                lake[self.state] = '@'
                # keep this state to use for rendering the next state in case it was the absorbing state
                self.last_state_rendered = lake
            else:
                # since the current state is the absorbing state, use the previous one for rendering
                lake = self.last_state_rendered

            print(lake.reshape(self.lake.shape))

        # render the given policy and value matrix
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            actions = ['^', 'v', '<', '>']

            # rendering the lake
            print('Lake:')
            print(self.lake)

            # rendering the policy
            print('Policy:')
            # converting numbers to arrows
            policy = np.array([actions[a] for a in policy[:-1]])

            # assigning holes and the goal to there coordinates
            policy = policy.reshape(self.lake.shape)
            for i in range(len(self.lake[0])):
                for j in range(len(self.lake[0])):
                    if self.lake[i][j] == '$':
                        print("$",end=" ")
                    elif self.lake[i][j] == '#':
                        print("#",end=" ")
                    else:
                        print(policy[i][j],end=" ")
                print('')

            # rendering the value matrix
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))