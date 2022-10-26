import gym
from SNNreinforcement import Agent
import matplotlib.pyplot as plt
import numpy as np
import time
from Producer import Producer
import pickle
import socket
socket.SO_REUSEADDR

#%%
# prod = Producer()
np.random.seed(42)
env=gym.make('LunarLander-v2', render_mode='rgb_array')
#env.seed(10)
lr=0.0005
agent=Agent(gamma=0.99, epsilon=0.0, lr=lr, input_dims=[8],n_actions=4, mem_size=1000000, batch_size=64, visualize=True)

agent.load_models()

compscores=0

agent.load_models()
print(agent.epsilon)
i=0
while True:
    done=False
    observation, _ = env.reset()
    score=0
    while not done:
        time.sleep(0.1)
        img = env.render()
        action, data, gaussX = agent.choose_action(observation)
 
        observation_, reward, done, info, _ = env.step(action)
        agent.store_transition(observation, action, reward, observation_, int(done))
        observation = observation_
        score += reward
    if i<5:
        compscores+=score
        print("Competitive round ",i+1," Overall score ",compscores)

    i+=1