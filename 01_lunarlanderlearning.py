import gym
from SNNreinforcement import Agent
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import torch

#%%
stime = time.time()
env = gym.make('LunarLander-v2', render_mode='rgb_array')

lr = 0.0005
n_games = 500
agent = Agent(gamma=0.99, epsilon=0.1, lr=lr, input_dims=[8], n_actions=4, 
              mem_size=1000000, batch_size=64, epsilon_end=0.01)

filename = "lunarlander.png"
scores = [] # Logged every episode
eps_history = [] # Logged every episode
avg_scores = [] # Logged every 10 episodes

score = 0
agent.load_models()

state = agent.q_net.state_dict()
state["lif1.beta"] = torch.tensor(0.95,dtype=torch.float)
state["lif2.beta"] = torch.tensor(0.95, dtype=torch.float)
state["lif3.beta"] = torch.tensor(1.0, dtype=torch.float)
agent.q_net.load_state_dict(state)
print(agent.q_net.lif1.beta)

#%%
i = -1
n_hrs = 1
while time.time() - stime < int(60 * 60 * n_hrs):
    i += 1
    done = False
    if i % 10 == 0 and i > 0:
        avg_score = np.mean(scores[max(0, i - 10):(i + 1)])
        avg_scores.append(avg_score)
        print('episode', i, 'score', score, 'average_score %.3f' % avg_score, 'epsilon %.3f' % agent.epsilon)
        agent.save_models()
    else:
        print('episode', i, 'score', score)

    observation, _ = env.reset()
    score = 0
    
    while not done:
        if i % 5 == 0:
            env.render()
        
        action = agent.choose_action(observation)
        observation_, reward, done, info, _ = env.step(action)
        score += reward

        agent.store_transition(observation, action, reward, observation_, int(done))
        observation = observation_
        agent.learn()

    scores.append(score)
    eps_history.append(agent.epsilon)

print("Total time: ", time.time() - stime)

x = [idx + 1 for idx in range(i + 1)]
plt.figure(0)
plt.plot(x, scores)
plt.grid()
plt.figure(1)
plt.plot(x, eps_history)
plt.grid()
plt.show()

dfname = str(datetime.now()).split('.')[0].replace(' ', '_').replace('-', '_').replace(':', '_') + '_runlogs'
df = pd.DataFrame({'score': scores, 'Epsilon': eps_history})
df_avg = pd.DataFrame({'avg_score': avg_scores})

df.to_csv(dfname+'.csv', sep=';', header=True, index=False)
df_avg.to_csv(dfname+'_avg.csv', sep=';', header=True, index=False)

print("Done logging results")

#%%
input("Press Enter to start trials\n")

compscores = 0
agent.load_models()

print('epsilon', agent.epsilon)

for i in range(5):
    done = False
    observation, _ = env.reset()
    score = 0
    while not done:
        env.render()
        action = agent.choose_action(observation)
        observation_, reward, done, info, _ = env.step(action)
        agent.store_transition(observation, action, reward, observation_, int(done))
        observation = observation_
        score += reward
    compscores += score
    print("Competitive round ", i + 1, " Overall score ", compscores)

with open("scoreboard.txt", "w") as f:
    f.writelines("%s: %i\n" % ("MIkro Lander", compscores))
input()
