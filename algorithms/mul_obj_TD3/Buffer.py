import numpy as np

class ReplayBuffer:
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

        return states, actions, rewards, new_states, termination

# import matplotlib.pyplot as plt

# alpha=0.000024
# beta='2'
# batch_size=64
# fig = plt.figure()
# fig.suptitle(t=f"{alpha} {batch_size}", fontsize=16)
# score_history=np.ones(3)
# TD_error=np.zeros(3)
#  # plot loss
# plt.subplot(2,2,1)
# plt.plot(score_history,'b')


# for i in range(len(running_avg)):
#     running_avg[i]=np.mean(score_history[max(0,i-100):(i+1)])
# plt.subplot(2,2,2)
# plt.plot(running_avg,'r')
# # lable='avg of previous 100 episodes'

# plt.subplot(2,2,3)
# plt.plot(TD_error,'g')

# plt.show()
