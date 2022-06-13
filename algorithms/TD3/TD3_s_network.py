import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir='./checkpoint/', device='cuda:0'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        # I think this breaks if the env has a 2D state representation
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.cs = nn.Linear(7, self.fc1_dims)
        normalized_cs = 1 / np.sqrt(self.cs.weight.data.size()[0])
        T.nn.init.uniform_(self.cs.weight.data, -normalized_cs, normalized_cs)
        T.nn.init.uniform_(self.cs.bias.data, -normalized_cs, normalized_cs)
        self.cs_bn1 = nn.LayerNorm(self.fc1_dims)

        self.cs2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        normalized_cs2 = 1 / np.sqrt(self.cs2.weight.data.size()[0])
        T.nn.init.uniform_(self.cs2.weight.data, -normalized_cs2, normalized_cs2)
        T.nn.init.uniform_(self.cs2.bias.data, -normalized_cs2, normalized_cs2)
        self.cs_bn2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(device)

    def forward(self, state, cs, action):
        cs_value = self.cs(cs)
        cs_value = self.cs_bn1(cs_value)
        cs_value = F.relu(cs_value)
        cs_value = self.cs2(cs_value)
        cs_value = self.cs_bn2(cs_value)

        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(T.add(q1_action_value, cs_value))
        # q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location="cuda:0"))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir='./checkpoint/', device='cuda:0'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.cs = nn.Linear(7, self.fc1_dims)
        normalized_cs = 1 / np.sqrt(self.cs.weight.data.size()[0])
        T.nn.init.uniform_(self.cs.weight.data, -normalized_cs, normalized_cs)
        T.nn.init.uniform_(self.cs.bias.data, -normalized_cs, normalized_cs)
        self.cs_bn1 = nn.LayerNorm(self.fc1_dims)

        self.cs2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        normalized_cs2 = 1 / np.sqrt(self.cs2.weight.data.size()[0])
        T.nn.init.uniform_(self.cs2.weight.data, -normalized_cs2, normalized_cs2)
        T.nn.init.uniform_(self.cs2.bias.data, -normalized_cs2, normalized_cs2)
        self.cs_bn2 = nn.LayerNorm(self.fc2_dims)

        self.ratio_dis = nn.Linear(1, 32)
        self.ratio_dis2 = nn.Linear(32, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(device)

    def forward(self, state, cs, ratio):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        cs_value = self.cs(cs)
        cs_value = self.cs_bn1(cs_value)
        cs_value = F.relu(cs_value)
        cs_value = self.cs2(cs_value)
        cs_value = self.cs_bn2(cs_value)
        cs_value = F.relu(cs_value)

        prob = T.add(prob, cs_value)

        y = self.ratio_dis(ratio)
        y = F.relu(y)
        y = self.ratio_dis2(y)
        y = F.relu(y)

        prob = T.tanh(T.sub(self.mu(prob), y))
        # prob = T.tanh(self.mu(prob))  # if action is > +/- 1 then multiply by max action

        return prob

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location="cuda:0"))
