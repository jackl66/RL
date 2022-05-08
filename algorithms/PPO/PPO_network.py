import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name,
                 chkpt_dir='checkpoint/'):
        super(ActorCritic, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name + '.zip')

        fc1 = 400
        fc2 = 300
        std = 0.05
        # ************************* critic ******************************#

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

        self.q = nn.Linear(fc2, 1)
        normalized_q = 0.003
        T.nn.init.uniform_(self.q.weight.data, -normalized_q, normalized_q)
        T.nn.init.uniform_(self.q.bias.data, -normalized_q, normalized_q)

        # ********************** actor ************************** #
        self.fc3 = nn.Linear(*input_dims, fc1)

        normalized_f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -normalized_f3, normalized_f3)
        T.nn.init.uniform_(self.fc3.bias.data, -normalized_f3, normalized_f3)
        self.bn3 = nn.LayerNorm(fc1)

        self.fc4 = nn.Linear(fc1, fc2)
        normalized_f4 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -normalized_f4, normalized_f4)
        T.nn.init.uniform_(self.fc4.bias.data, -normalized_f4, normalized_f4)
        self.bn4 = nn.LayerNorm(fc2)

        self.mu = nn.Linear(fc2, n_actions)
        normalized_mu = 0.003
        T.nn.init.uniform_(self.mu.weight.data, -normalized_mu, normalized_mu)
        T.nn.init.uniform_(self.mu.bias.data, -normalized_mu, normalized_mu)

        self.log_std = nn.Parameter(T.ones(n_actions) * std)

        # ********************** visual input ************************** #
        # use two dense layer to learn the depth matrix as well
        self.cs = nn.Linear(7, fc1)
        normalized_depth = 1 / np.sqrt(self.cs.weight.data.size()[0])
        T.nn.init.uniform_(self.cs.weight.data, -normalized_depth, normalized_depth)
        T.nn.init.uniform_(self.cs.bias.data, -normalized_depth, normalized_depth)
        self.cs_bn1 = nn.LayerNorm(fc1)

        self.cs2 = nn.Linear(fc1, fc2)
        normalized_depth2 = 1 / np.sqrt(self.cs2.weight.data.size()[0])
        T.nn.init.uniform_(self.cs2.weight.data, -normalized_depth2, normalized_depth2)
        T.nn.init.uniform_(self.cs2.bias.data, -normalized_depth2, normalized_depth2)
        self.cs_bn2 = nn.LayerNorm(fc2)

        # **************  surface to volume ratio ***********************#
        self.ratio_dis = nn.Linear(1, 32)
        self.ratio_dis2 = nn.Linear(32, n_actions)

        self.dropout = nn.Dropout(0.2)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state, cross_section, ratio):
        # **************  surface to volume ratio ***********************#
        y = self.ratio_dis(ratio)
        y = F.relu(y)
        y = self.ratio_dis2(y)
        y = F.relu(y)

        # ************************* visual ******************************#
        x = self.cs(cross_section)
        x = self.cs_bn1(x)
        x = F.relu(x)
        x = self.cs2(x)
        x = self.cs_bn2(x)

        # ************************* critic ******************************#

        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        # state_value = self.dropout(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        # print(state_value.size(),x.size())
        # state_value = self.dropout(state_value)
        state_value = F.relu(T.add(state_value, x))
        state_value = self.q(state_value)

        # ************************* actor ******************************#
        mu = self.fc3(state)
        mu = self.bn3(mu)
        mu = F.relu(mu)
        mu = self.fc4(mu)
        mu = self.bn4(mu)

        # state_value = self.dropout(state_value)
        mu = T.add(mu, x)
        mu = T.tanh(T.add(self.mu(mu),y))
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist, state_value

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


