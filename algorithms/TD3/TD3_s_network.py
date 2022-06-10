import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, name,
                 chkpt_dir='checkpoint/td3', device='cuda:0'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name + '.zip')

        fc1 = 400
        fc2 = 300
        # ************************* Q1 ******************************#

        self.fc1 = nn.Linear(*input_dims, fc1)

        normalized_f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -normalized_f1, normalized_f1)
        T.nn.init.uniform_(self.fc1.bias.data, -normalized_f1, normalized_f1)
        self.bn1 = nn.LayerNorm(fc1)

        self.fc2 = nn.Linear(fc1, fc2)
        normalized_f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -normalized_f2, normalized_f2)
        T.nn.init.uniform_(self.fc2.bias.data, -normalized_f2, normalized_f2)
        self.bn2 = nn.LayerNorm(fc2)

        self.action_value = nn.Linear(n_actions, fc2)

        self.q = nn.Linear(fc2, 1)
        normalized_q = 0.003
        T.nn.init.uniform_(self.q.weight.data, -normalized_q, normalized_q)
        T.nn.init.uniform_(self.q.bias.data, -normalized_q, normalized_q)

        # ************************* Q2 ******************************#

        self.fc3 = nn.Linear(*input_dims, fc1)

        normalized_f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -normalized_f3, normalized_f3)
        T.nn.init.uniform_(self.fc3.bias.data, -normalized_f3, normalized_f3)
        self.bn3 = nn.LayerNorm(fc1)

        self.fc4 = nn.Linear(fc1, fc2)
        normalized_f4 = 1 / np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -normalized_f4, normalized_f4)
        T.nn.init.uniform_(self.fc4.bias.data, -normalized_f4, normalized_f4)
        self.bn4 = nn.LayerNorm(fc2)

        self.action_value2 = nn.Linear(n_actions, fc2)

        self.q2 = nn.Linear(fc2, 1)
        normalized_q2 = 0.003
        T.nn.init.uniform_(self.q2.weight.data, -normalized_q2, normalized_q2)
        T.nn.init.uniform_(self.q2.bias.data, -normalized_q2, normalized_q2)

        # use two dense layer to learn the cs matrix as well
        self.cs = nn.Linear(7, fc1)
        normalized_cs = 1 / np.sqrt(self.cs.weight.data.size()[0])
        T.nn.init.uniform_(self.cs.weight.data, -normalized_cs, normalized_cs)
        T.nn.init.uniform_(self.cs.bias.data, -normalized_cs, normalized_cs)
        self.cs_bn1 = nn.LayerNorm(fc1)

        self.cs2 = nn.Linear(fc1, fc2)
        normalized_cs2 = 1 / np.sqrt(self.cs2.weight.data.size()[0])
        T.nn.init.uniform_(self.cs2.weight.data, -normalized_cs2, normalized_cs2)
        T.nn.init.uniform_(self.cs2.bias.data, -normalized_cs2, normalized_cs2)
        self.cs_bn2 = nn.LayerNorm(fc2)

        self.dropout = nn.Dropout(0.2)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(device)

    def forward(self, state, cs, action):
        cs_value = self.cs(cs)
        cs_value = self.cs_bn1(cs_value)
        cs_value = F.relu(cs_value)
        cs_value = self.cs2(cs_value)
        cs_value = self.cs_bn2(cs_value)

        # ************************* Q1 ******************************#

        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.dropout(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        state_value = self.dropout(state_value)
        state_value = F.relu(T.add(state_value, cs_value))

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        # ************************* Q2 ******************************#
        # state_value = self.dropout(state_value)

        state_value2 = self.fc3(state)
        state_value2 = self.bn3(state_value2)
        state_value2 = F.relu(state_value2)
        state_value2 = self.fc4(state_value2)
        state_value2 = self.bn4(state_value2)

        # state_value = self.dropout(state_value)

        state_value2 = F.relu(T.add(state_value2, cs_value))

        action_value2 = F.relu(self.action_value2(action))
        state_action_value2 = F.relu(T.add(state_value2, action_value2))
        state_action_value2 = self.q(state_action_value2)

        return state_action_value, state_action_value2

    def Q1(self, state, cs, action):
        cs_value = self.cs(cs)
        cs_value = self.cs_bn1(cs_value)
        cs_value = F.relu(cs_value)
        cs_value = self.cs2(cs_value)
        cs_value = self.cs_bn2(cs_value)


        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        # state_value = self.dropout(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        # state_value = self.dropout(state_value)

        state_value = F.relu(T.add(state_value, cs_value))

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file,map_location="cuda:0"))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name,
                 chkpt_dir='checkpoint/td3', device='cuda:0'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name + '.zip')
        fc1 = 400
        fc2 = 300

        self.fc1 = nn.Linear(*input_dims, fc1)

        normalized_f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -normalized_f1, normalized_f1)
        T.nn.init.uniform_(self.fc1.bias.data, -normalized_f1, normalized_f1)
        self.bn1 = nn.LayerNorm(fc1)

        self.fc2 = nn.Linear(fc1, fc2)
        self.bn2 = nn.LayerNorm(fc2)
        normalized_f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -normalized_f2, normalized_f2)
        T.nn.init.uniform_(self.fc2.bias.data, -normalized_f2, normalized_f2)

        self.mu = nn.Linear(fc2, n_actions)
        normalized_mu = 0.003
        T.nn.init.uniform_(self.mu.weight.data, -normalized_mu, normalized_mu)
        T.nn.init.uniform_(self.mu.bias.data, -normalized_mu, normalized_mu)

        # use two dense layer to learn the cs matrix as well
        self.cs = nn.Linear(7, fc1)
        normalized_cs = 1 / np.sqrt(self.cs.weight.data.size()[0])
        T.nn.init.uniform_(self.cs.weight.data, -normalized_cs, normalized_cs)
        T.nn.init.uniform_(self.cs.bias.data, -normalized_cs, normalized_cs)
        self.cs_bn1 = nn.LayerNorm(fc1)

        self.cs2 = nn.Linear(fc1, fc2)
        normalized_cs2 = 1 / np.sqrt(self.cs2.weight.data.size()[0])
        T.nn.init.uniform_(self.cs2.weight.data, -normalized_cs2, normalized_cs2)
        T.nn.init.uniform_(self.cs2.bias.data, -normalized_cs2, normalized_cs2)
        self.cs_bn2 = nn.LayerNorm(fc2)

        self.ratio_dis = nn.Linear(1, 32)
        self.ratio_dis2 = nn.Linear(32, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.dropout = nn.Dropout(0.2)

        self.to(device)

    def forward(self, state, cs, ratio):

        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout(x)

        cs_value = self.cs(cs)
        cs_value = self.cs_bn1(cs_value)
        cs_value = F.relu(cs_value)
        cs_value = self.cs2(cs_value)
        cs_value = self.cs_bn2(cs_value)
        cs_value = F.relu(cs_value)

        x = T.add(x, cs_value)

        y = self.ratio_dis(ratio)
        y = F.relu(y)
        y = self.ratio_dis2(y)
        y = F.relu(y)
        # x = T.tanh(T.sub(self.mu(x), y))
        x = T.tanh(T.add(self.mu(x), y))

        # x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file,map_location="cuda:0"))
