from environment.frozen_lake import FrozenLake 


seed = 0
# Small lake
lake =   [['&', '.', '.', '.'],
          ['.', '#', '.', '#'],
          ['.', '.', '.', '#'],
          ['#', '.', '.', '$']]

env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
done = False 
while not done:
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            env.render()
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))
        print(done)
        env.render()
        print('Reward: {0}.'.format(r))