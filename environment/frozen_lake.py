#!/usr/bin/env python3

import numpy as np
from .environment import Environment, _printoptions

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
        
        # TODO:
        Environment.__init__(self, n_states, n_actions, max_steps, None)
        # Load the precomputed transition probabilities
        self._p = np.load('p.npy')
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def p(self, next_state, state, action):
        # TODO:
        return self._p[next_state, state, action]

    def r(self, next_state, state, action):
        # TODO:
        if state < self.n_states - 1 and self.lake_flat[state] == '$':
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
            # actions = ['^', '<', '_', '>']
            actions = ['↑', '←', '↓', '→']


            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])

            # We have to do add this to the end so printed policy does not overwrite the holes and goal tile with actions arrows
            policy = policy.reshape(self.lake.shape)
            for i in range(len(self.lake[0])):
                for j in range(len(self.lake[0])):
                    if self.lake[i][j] == '$':
                        policy[i][j] = '$'
                    elif self.lake[i][j] == '#':
                        policy[i][j] = '#'
                    else:
                        pass
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))