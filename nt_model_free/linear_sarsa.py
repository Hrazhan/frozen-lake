import numpy as np


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO:
        is_terminal = False
        if np.random.rand() < (1 - epsilon[i]) and max(q) != 0:
            a = np.argmax(q)
        else:
            a = np.random.choice(env.n_actions)
        while not is_terminal:
            nxt_feat, reward, is_terminal = env.step(a)
            delta_val = reward - q[a]
            q = nxt_feat.dot(theta)
            if np.random.rand() < (1 - epsilon[i]) and max(q) != 0:
                nxt_act = np.argmax(q)
            else:
                nxt_act = np.random.choice(env.n_actions)
            delta_val += gamma * q[nxt_act]
            theta += eta[i] * delta_val * features[a]
            features = nxt_feat
            a = nxt_act
    return theta
