################ Environment ################

import numpy as np
from FrozenLake import FrozenLake
import time


def play(env):
    # Up, down, left, right, stay
    actions = ['w', 's', 'a', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            env.render()
            continue
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))
        env.render()
        print('Reward: {0}.'.format(r))

################ Model-based algorithms ################
def action_value(env, value, state, action, gamma):
    return sum([env.p(next_s, state, action) * (env.r(next_s, state, action) + gamma * value[next_s]) for next_s in range(env.n_states)])

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = action_value(env, value, s, policy[s], gamma)

            delta = max(delta, abs(v - value[s]))

        if delta < theta:
            break

    return value

def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    for s in range(env.n_states):
        policy[s] = np.argmax([action_value(env, value, s, a, gamma) for a in range(env.n_actions)])

    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    for i in range(max_iterations):
        values = policy_evaluation(env, policy, gamma, theta,max_iterations)
        policy = policy_improvement(env, values, gamma)

    return policy, values
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    for _ in range(max_iterations):
        delta = 0
        
        for s in range(env.n_states):
            v = value[s]
            value[s] = max([action_value(env, value, s, a, gamma) for a in range(env.n_actions)])
    
            delta = max(delta, np.abs(v - value[s]))

        if delta < theta:
            break

    policy = policy_improvement(env, value, gamma)

    return policy, value

################ Tabular model-free algorithms ################

# Selecting an action for state s according to e-greedy policy based on q
def EGreedySelection(env, random_state, q, s, epsilon):
    if s < env.lake.size:
        actions = env._possible_actions[s]
    else:
        actions = range(env.n_actions)

    if random_state.rand() < epsilon:
        return random_state.choice(actions)
    else:
        return actions[np.argmax(q[actions])]

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()

        a = EGreedySelection(env, random_state, q[s], s, epsilon[i])
        done = False
        while not done:
            next_s, r, done = env.step(a)
            next_a = EGreedySelection(env, random_state, q[next_s], next_s, epsilon[i])
            q[s, a] = q[s, a] + eta[i] * (r + gamma * q[next_s, next_a] - q[s, a])
            s = next_s
            a = next_a
        print('%d/%d' % (i, max_episodes))
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value
    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None, render=False):
    random_state = EGreedySelection(epsilon)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    q.fill(float('-inf'))
    
    for s in range(env.n_states - 1):
        actions = env._possible_actions[s]
        for a in actions:
            q[s, a] = 0

    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        done = False
        while not done:
            a = random_state(q[s], env._possible_actions[s])
            
            next_s, r, done = env.step(a)
            
            # when reaching absorbing state use same action
            if done:
                q[s,a] = q[s,a] + eta[i] * (r - q[s,a ])
            else:
                q[s, a] = q[s, a] + eta[i] * (r + gamma * max(q[next_s]) - q[s, a])

            s = next_s

    if render:
        run_agent(q,env,epsilon=0) # High Exploration 
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value

################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = EGreedySelection(epsilon)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    for s in range(env.n_states):
        actions = env._possible_actions[s]
        for a in actions:
            q[s, a] = 0


    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO:
    
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO:

    return theta    

################ Main function ################

def main():
    seed = 0
    
    # Small lake
    smallLake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],    
            ['#', '.', '.', '$']]
    
    # Large lake
    largeLake = [['&', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '#', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '#', '.'], 
            ['.', '.', '.', '.', '#', '.', '.', '.'], 
            ['.', '.', '.', '#', '.', '.', '.', '.'], 
            ['.', '#', '#', '.', '.', '.', '#', '.'], 
            ['.', '#', '.', '.', '#', '.', '#', '.'],   
            ['.', '.', '.', '#', '.', '.', '.', '$']]
    lake = largeLake
    env = FrozenLake(lake, slip=0.1, max_steps=np.array(lake).size, seed=seed)
    
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 15


    # print('## Play')
    # play(env)
    # return
    # print('')
    
    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    print('')
    
    print('# Model-free algorithms')
    max_episodes = 50000
    eta = 0.5
    epsilon = 0.5
    
    print('')
    
    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    return
    print('')
    
    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    return
    linear_env = LinearWrapper(env)
    
    print('## Linear Sarsa')
    
    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    
    print('')
    
    print('## Linear Q-learning')
    
    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                   gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

def run_agent(q, env,epsilon):
    for test in range(20):
        print(f"Test #{test}")
        s = env.reset()
        random_state = EGreedySelection(epsilon)
        total_reward = 0
        done = False
        while True:
            time.sleep(0.5)
            env.render()
            a = random_state(q[s], env._possible_actions[s])
            print(f"Chose action {a} for state {s}")
            s, reward, done = env.step(a)
            total_reward += reward
            if done:
                print(f"Episode reward: {total_reward}")
                time.sleep(1)
                break

if __name__ == "__main__":
    main()
