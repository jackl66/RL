import numpy as np


class oneD_ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.termination_memory = np.zeros(self.mem_size, dtype=int)
        self.mem_cntr = 0



    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.termination_memory[index] = done
        self.action_memory[index] = action



        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        termination = self.termination_memory[batch]
        #
        # cross_section = self.cross_section_memory[batch]
        # new_cross_section = self.new_cross_section_memory[batch]

        return states, actions, rewards, new_states, termination


