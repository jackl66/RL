import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np


class DQN(nn.Module):
    def __init__(self, beta, input_dims, n_actions, name,
                 chkpt_dir='checkpoint/DQN/', device='cuda:0'):
        super(DQN, self).__init__()

        # os.mkdir(chkpt_dir) done in the actor network
        self.checkpoint_file = os.path.join(chkpt_dir, name + '.zip')

        fc1 = 400
        fc2 = 300

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

        self.q = nn.Linear(fc2, n_actions)
        normalized_q = 0.003
        T.nn.init.uniform_(self.q.weight.data, -normalized_q, normalized_q)
        T.nn.init.uniform_(self.q.bias.data, -normalized_q, normalized_q)


        # use two dense layer to learn the depth matrix as well
        self.depth = nn.Linear(16 * 16, fc1)
        normalized_depth = 1 / np.sqrt(self.depth.weight.data.size()[0])
        T.nn.init.uniform_(self.depth.weight.data, -normalized_depth, normalized_depth)
        T.nn.init.uniform_(self.depth.bias.data, -normalized_depth, normalized_depth)
        self.depth_bn1 = nn.LayerNorm(fc1)

        self.depth2 = nn.Linear(fc1, fc2)
        normalized_depth2 = 1 / np.sqrt(self.depth2.weight.data.size()[0])
        T.nn.init.uniform_(self.depth2.weight.data, -normalized_depth2, normalized_depth2)
        T.nn.init.uniform_(self.depth2.bias.data, -normalized_depth2, normalized_depth2)
        self.depth_bn2 = nn.LayerNorm(fc2)
        # use two dense layer to learn the depth matrix as well
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
        self.dropout = nn.Dropout(0.5)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(device)

    def forward(self, state, depth,cs):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        depth_value = self.depth(depth)
        depth_value = self.depth_bn1(depth_value)
        depth_value = F.relu(depth_value)
        depth_value = self.depth2(depth_value)
        depth_value = self.depth_bn2(depth_value)
        # state_value=self.dropout(state_value)
        cs_value = self.cs(cs)
        cs_value = self.cs_bn1(cs_value)
        cs_value = F.relu(cs_value)
        cs_value = self.cs2(cs_value)
        cs_value = self.cs_bn2(cs_value)
         
        state_value = F.relu(T.add(state_value, cs_value))
        state_action_value = self.q(T.add(depth_value, state_value))

        return state_action_value

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
