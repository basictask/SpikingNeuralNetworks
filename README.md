# SpikingNeuralNetworks
Spiking Deep Learning for Deep Reinforcement learning
## Made by Daniel Kuknyo
This part of the project is aimed at continuous control: the environments used here were MountainCarContinuous-v0 and BipedalWalker-v3.
In continuous control the prediction is aimed not at Q-values as there is no discrete action space available for the agent. 
For BipedalWalker the action space is a 4-element vector. Each vector represents an angle for the corresponding joint of the BipedalWalker. The state space is a 24-element vector. 
The goal of the agent is to walk to the destination on an uneven surface. 
### Folder structure
1. models: trained model weights for the different models
2. notebooks: implemented algorithms in Jupyter Notebook format
3. results: running logs for learning: epsilon history, rewards and mean rewards
4. scoreboards: scoring for competitive running in csv files

