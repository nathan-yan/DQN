import theano
import theano.tensor as T

import gym

import numpy as np
import matplotlib.pyplot as plt

from six.moves import cPickle

from dqn import DQN, Approximator

def preprocess(observation):
    return np.expand_dims(observation, axis = 0)

def main():
    target = Approximator(observation_type = T.matrix())
    approximator = Approximator(observation_type = T.matrix())
    dqn = DQN(network = approximator)

    env = gym.make("LunarLander-v2")

    # Initialize target approximator with current
    dqn.copy_weights(target)

    total_ts = 0

    episodes = 100000
    for episode in range (episodes):
        done = False
        ts = 0
        r = 0

        observation = preprocess(env.reset())
        while not done and ts < 5000:
            (observation, reward, done, info), _ = dqn.step(env, observation, preprocess)
            observation = preprocess(observation)

            r += reward

            if episode % 10 == 0:
                env.render()

            if total_ts % 1000 == 0:
                # Update target network
                #print(target.get_weights()[0].get_value())
                dqn.copy_weights(target)
                #print(target.get_weights()[0].get_value())

            if total_ts % 10000 == 0:
                print("SAVING WEIGHTS")
                weight = open("dqn_lunarlander" + str(total_ts) + ".w", 'wb')
                for w in dqn.network.get_weights():
                    cPickle.dump(w, weight, protocol = cPickle.HIGHEST_PROTOCOL)

                weight.close()

            if (len(dqn.memory) > 2000 and total_ts % 5 == 0):
                dqn.train(target)

            total_ts += 1
            ts += 1

        print("Reward: " + str(r) + " | Epsilon: " + str(dqn.epsilon))

if __name__ == "__main__":
    main()