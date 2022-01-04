from time import process_time
import numpy as np
from environment.frozen_lake import FrozenLake
from model_based.policy_iteration import policy_iteration
from model_based.value_iteration import value_iteration

from model_based.policy_iteration import policy_evaluation 

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    eps_count = 0
    _, optimal_policy = policy_iteration(env, 0.9, 0.001, 100)
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
    
        # select an action based on E-greedy 
        if np.random.random() <= epsilon[i]:
            a = np.random.randint(0, env.n_actions)
        else:
            a = np.argmax(q[s,:])

        done = False
        while not done:
            n_s, reward, done = env.step(a)

            # Select new action using e greedy for next state
            if np.random.random() <= epsilon[i]:
                next_action = np.random.randint(0, env.n_actions)
            else:
                next_action = np.argmax(q[n_s,:])

            q[s, a] = q[s, a] + (eta[i] * (reward + (gamma * q[n_s, next_action]) - q[s, a]))
            s = n_s
            a = next_action
            
        eps_count += 1
        if np.all(np.abs(optimal_policy - q.max(axis=1)) < 0.1):
            print("eps_count", eps_count)
            break
    # value_temp = policy_evaluation(env, q.argmax(axis=1) ,gamma, 0.001, 100)
    # print(value_temp)
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value

def run():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]

    # Big lake
    big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                  ['.', '.', '.', '.', '.', '.', '.', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '.'],
                  ['.', '.', '.', '.', '.', '#', '.', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '.'],
                  ['.', '#', '#', '.', '.', '.', '#', '.'],
                  ['.', '#', '.', '.', '#', '.', '#', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '$']]

    lake = big_lake
    env = FrozenLake(lake, slip=0.1, max_steps=64, seed=seed)

    gamma = 0.9
    theta = 0.001
    max_iterations = 100


    print('## Measuring Policy iteration for a 100 run')
    durations = []

    for i in range(100):
        start = process_time()
        policy, value = policy_iteration(env, gamma, theta, max_iterations)


        end = process_time()
        duration = end - start
        durations.append(duration)

    print("Mean time:", round(np.mean(durations), 4), "seconds")

    print("\n\n")
    print('## Measuring Value iteration for a 100 run')
    durations = []

    for i in range(100):
        start = process_time()
        policy, value = value_iteration(env, gamma, theta, max_iterations)


        end = process_time()
        duration = end - start
        durations.append(duration)

    print("Mean time:", round(np.mean(durations), 4), "seconds")


if __name__ == "__main__":
    run()
