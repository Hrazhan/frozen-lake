from time import process_time
import numpy as np
from environment.frozen_lake import FrozenLake
from model_based.policy_iteration import policy_iteration
from model_based.value_iteration import value_iteration


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
