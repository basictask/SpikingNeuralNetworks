import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta=0.95):
        super(Net, self).__init__()

        spike_grad1 = surrogate.fast_sigmoid()
        spike_grad2 = surrogate.fast_sigmoid()
        spike_grad3 = surrogate.fast_sigmoid()
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, learn_threshold=False, spike_grad=spike_grad1)
        print(self.lif1)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, learn_threshold=False, spike_grad=spike_grad2)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = snn.Leaky(
            beta=beta,
            learn_beta=True,
            threshold=1e5,
            reset_mechanism="none",
            spike_grad=spike_grad3,
        )

    def _convert_to_spikes(self, data):
        return snn.spikegen.delta(data, threshold=0.1, padding=False, off_spike=True)

    def createGauss(self, mins, maxes, numPerDim, amplMax, dims):
        self.amplMax = amplMax
        self.numPerDim = numPerDim
        self.M = []
        self.sigma = []
        for i in range(dims):
            M, sigma = np.linspace(mins[i], maxes[i], numPerDim, retstep=True)
            self.M.append(M)
            self.sigma += [
                sigma,
            ] * self.numPerDim
        self.M = torch.tensor(
            np.array(self.M).reshape(-1, self.numPerDim), dtype=torch.float
        ).cuda()
        self.sigma = torch.tensor(
            np.array(self.sigma).reshape(-1, self.numPerDim), dtype=torch.float
        ).cuda()

    def gaussianCurrents(self, data):
        x = data.unsqueeze(-1).repeat([1, 1, self.numPerDim])
        return (
            torch.exp(-1 / 2 * ((x - self.M) / self.sigma) ** 2) * self.amplMax
        ).reshape(data.shape[0], -1)

    def forward(self, x, num_steps=16):

        x = self.gaussianCurrents(x)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        spk2_rec = []
        mem2_rec = []

        spk1_rec = []
        mem1_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(
                cur1,
                mem1,
            )
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

        return (
            x,
            [
                torch.stack(spk1_rec, dim=0),
                torch.stack(spk2_rec, dim=0),
                torch.stack(spk3_rec, dim=0),
                torch.stack(mem1_rec, dim=0),
                torch.stack(mem2_rec, dim=0),
                torch.stack(mem3_rec, dim=0),
            ],
            torch.stack(mem3_rec, dim=0),
        )
        # '''


class Agent(object):
    def __init__(
        self,
        lr,
        gamma,
        mem_size,
        n_actions,
        epsilon,
        batch_size,
        input_dims,
        epsilon_dec=0.99988,
        epsilon_end=0.01,
        targetUpdateSteps=20,
        q_dir="tmp\\q",
        visualize=False,
    ):
        self.visualize = visualize
        self.targetUpdateSteps = targetUpdateSteps
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.stepnum = 0
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size

        self.low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5,
                -1.5,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -np.pi,
                -5.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        self.high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5,
                1.5,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                np.pi,
                5.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        self.dtype = torch.float
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        gaussPerDim = 64
        hiddenSize = 128
        self.q_net = Net(input_dims[0] * gaussPerDim, hiddenSize, n_actions).to(
            self.device
        )
        self.q_net.createGauss(self.low, self.high, gaussPerDim, 1.0, input_dims[0])
        self.t_net = Net(input_dims[0] * gaussPerDim, hiddenSize, n_actions).to(
            self.device
        )
        self.t_net.createGauss(self.low, self.high, gaussPerDim, 1.0, input_dims[0])
        self.t_net.load_state_dict(self.q_net.state_dict())

        # self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=lr, betas=(0.9, 0.999)
        )
        # self.q_eval = DeepQNetwork(lr, n_actions, input_dims=input_dims, name='q_eval', chkpt_dir=q_dir)

        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.terminal_memory[index] = 1 - terminal
        self.mem_cntr += 1

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        # print(self.epsilon)
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            with torch.no_grad():
                self.q_net.eval()
                gaussX, spikes, actions = self.q_net(
                    torch.tensor(state, dtype=self.dtype).to(self.device)
                )
                actions = actions.cpu()[-1, 0, :].detach().numpy()
                action = np.argmax(actions)
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:
            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size

            batch = np.random.choice(max_mem, self.batch_size)
            state_batch = self.state_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]

            self.q_net.train()
            evalspikes, _, q_eval = self.q_net.forward(
                torch.tensor(state_batch, dtype=self.dtype).to(self.device)
            )
            q_eval = q_eval[-1, :, :]

            self.t_net.eval()
            _, _, q_next = self.t_net.forward(
                torch.tensor(new_state_batch, dtype=self.dtype).to(self.device)
            )

            q_next = q_next[-1, :, :]

            q_target = torch.clone(q_eval).detach()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = (
                (
                    torch.tensor(reward_batch, device=self.device)
                    + torch.tensor(self.gamma, device=self.device)
                    * torch.max(q_next, dim=1).values
                    * torch.tensor(terminal_batch, device=self.device)
                )
                .type(self.dtype)
                .detach()
            )

            criterion = nn.MSELoss()
            loss = criterion(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()

            for param in self.q_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            self.epsilon = (
                self.epsilon * self.epsilon_dec
                if self.epsilon > self.epsilon_end
                else self.epsilon_end
            )

            if self.stepnum % self.targetUpdateSteps == 0:
                self.t_net.load_state_dict(self.q_net.state_dict())

            self.stepnum += 1

    def save_models(self):
        torch.save(self.q_net.state_dict(), "./tmp/dict9 LIF cartpole")

    def load_models(self):
        self.q_net.load_state_dict(torch.load("./tmp/dict9 LIF cartpole"))
        self.q_net.eval()
