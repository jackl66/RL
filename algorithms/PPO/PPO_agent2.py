import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.cross_sections = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.cross_sections), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, cross_section, action, probs, vals, reward, done):
        self.states.append(state)
        self.cross_sections.append(cross_section)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.cross_sections = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []




class Agent2:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=2, n_epochs=6):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, cross_section, action, probs, vals, reward, done):
        self.memory.store_memory(state, cross_section, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, cross_section, ratio):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        cross_section = T.tensor([cross_section], dtype=T.float).to(self.actor.device)
        dist = self.actor(state, cross_section, ratio)
        value = self.critic(state, cross_section)
        action = dist.sample()

        probs = np.squeeze(dist.log_prob(action).cpu().detach().numpy())
        action = np.squeeze(action.cpu().detach().numpy())
        value = np.squeeze(value.cpu().detach().numpy())

        return action, probs, value

    def learn(self, ratio):
        for _ in range(self.n_epochs):
            state_arr, cross_section_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage, requires_grad=True).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                cross_section = T.tensor(cross_section_arr[batch], dtype=T.float).to(self.actor.device)

                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states, cross_section, ratio)
                critic_value = self.critic(states, cross_section)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                print(f'{states.size()},{cross_section.size()},{prob_ratio.size()}\n{advantage[batch].size()},'
                      f'{critic_value.size()}')
                print(advantage[batch], '\n', prob_ratio)
                ad = advantage[batch]

                weighted_probs = prob_ratio * ad
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * ad
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
