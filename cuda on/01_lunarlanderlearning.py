import gym

from SNNreinforcement import Agent

import matplotlib.pyplot as plt
import numpy as np
import time
import torch

if __name__ == "__main__":

    # np.random.seed(1233)

    stime = time.time()
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    # env.seed(42)
    lr = 0.0005
    n_games = 500
    agent = Agent(
        gamma=0.99,
        epsilon=0.1,
        lr=lr,
        input_dims=[8],
        n_actions=4,
        mem_size=1000000,
        batch_size=64,
        epsilon_end=0.01,
    )

    filename = "lunarlander.png"
    scores = []
    eps_history = []

    score = 0
    agent.load_models()

    state = agent.q_net.state_dict()
    state["lif1.beta"] = torch.tensor(0.95, dtype=torch.float).cuda()
    state["lif2.beta"] = torch.tensor(0.95, dtype=torch.float).cuda()
    state["lif3.beta"] = torch.tensor(1.0, dtype=torch.float).cuda()
    agent.q_net.load_state_dict(state)
    print(agent.q_net.lif1.beta)

    i = -1
    while time.time() - stime < 60 * 240:
        i += 1
        done = False
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i - 10) : (i + 1)])
            print(
                "episode",
                i,
                "score",
                score,
                "average_score %.3f" % avg_score,
                "epsilon %.3f" % agent.epsilon,
            )
            agent.save_models()
        else:
            print("episode", i, "score", score)

        observation, _ = env.reset()
        score = 0
        while not done:
            if i % 5 == 0:
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward

            # print(score, reward)
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = observation_

            agent.learn()

        scores.append(score)
        eps_history.append(agent.epsilon)

    x = [idx + 1 for idx in range(i + 1)]
    plt.figure(0)
    plt.plot(x, scores)
    plt.grid()
    plt.figure(1)
    plt.plot(x, eps_history)
    plt.grid()
    print("Total time: ", time.time() - stime)
    plt.show()
    # '''
    input("Press Enter to start trials\n")

    compscores = 0

    agent.load_models()
    print(agent.epsilon)
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
