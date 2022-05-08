import numpy as np


class Vision_ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.state_memory = np.zeros((self.mem_size, 7))
        self.new_state_memory = np.zeros((self.mem_size, 7))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.termination_memory = np.zeros(self.mem_size, dtype=int)
        self.mem_cntr = 0

        self.depth_memory = np.zeros((self.mem_size, 256))
        self.new_depth_memory = np.zeros((self.mem_size, 256))

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state[:-1]
        self.new_state_memory[index] = new_state[:-1]
        self.reward_memory[index] = reward
        self.termination_memory[index] = done
        self.action_memory[index] = action

        # flatten the 9*9 depth matrix
        flatten_depth = np.array(state[-1])
        new_flatten_depth = np.array(new_state[-1])

        self.depth_memory[index] = flatten_depth.ravel()
        self.new_depth_memory[index] = new_flatten_depth.ravel()

        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        termination = self.termination_memory[batch]

        depth = self.depth_memory[batch]
        new_depth = self.new_depth_memory[batch]

        return states, actions, rewards, new_states, depth, new_depth, termination
