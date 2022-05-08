from ..replay_buffer.vision_Buffer import Vision_ReplayBuffer as ReplayBuffer
from .vision_DQN_network import *
import math


class Vision_DQN_agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99,
                 n_actions=2, max_size=1000000, batch_size=64, token=0, update_freq=4, idx='0', eval=0):
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.count = 0
        self.step = 0
        cuda_idx = 'cuda:' + idx
        self.device = T.device(cuda_idx if T.cuda.is_available() else 'cpu')
        chkpt_dir = './checkpoint/DQN/' + str(token)
        self.noise_clip = 0.5
        self.mask = T.ones(self.batch_size).to(self.device)
        if eval == 0:
            os.mkdir(chkpt_dir)

        self.Qnet = DQN(beta, input_dims, n_actions,
                        'Qnet', chkpt_dir=chkpt_dir, device=self.device)

        self.target_Qnet = DQN(beta, input_dims, n_actions,
                               'target_Qnet', chkpt_dir=chkpt_dir, device=self.device)

        eps_start = 1
        eps_end = 0.01
        eps_decay = 0.001
        self.strategy = Eps(eps_start, eps_end, eps_decay)
        self.update_network_parameters()

    def choose_action(self, observation):

        rate = self.strategy.get_explor_rate(self.step)
        self.step += 1
        if rate > np.random.rand():
            return np.random.randint(9)   # explore
        else:
            with T.no_grad():
                observation1 = observation[:-2]

                depth = observation[-1]
                depth = np.ravel(depth)
                depth = T.tensor(depth, dtype=T.float).to(self.device)
                cs = observation[-2]
                cs = np.ravel(cs)
                cs = T.tensor(cs, dtype=T.float).to(self.device)
                observation = T.tensor(observation1, dtype=T.float).to(self.device)

                action = self.Qnet.forward(observation, depth,cs).to(self.device)  # exploit
                action = action.cpu().detach().numpy()
                idx = np.argmax(action)
                return idx

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return "nope", "try again"

        state, action, reward, new_state, depth, new_depth, cs, new_cs, done = self.memory.sample(self.batch_size)

         
        new_state = T.tensor(new_state, dtype=T.float).to(self.device)
       
        state = T.tensor(state, dtype=T.float).to(self.device)
        depth = T.tensor(depth, dtype=T.float).to(self.device)
        new_depth = T.tensor(new_depth, dtype=T.float).to(self.device)
        # reward = T.tensor(reward, dtype=T.float).to(self.device)
        # done = T.tensor(done).to(self.device)
        cs = T.tensor(cs, dtype=T.float).to(self.device)
        new_cs = T.tensor(new_cs, dtype=T.float).to(self.device)

        # action = T.tensor(action, dtype=T.float).to(self.device)
        # Get current Q estimates
        current_Q1 = self.Qnet.forward(state, depth,cs)
        
        # Compute critic loss
        # masked = T.sub(self.mask, done)
        # masked = T.reshape(masked, (self.batch_size, 1))

        target_Q = self.target_Qnet.forward(new_state, new_depth,new_cs)
        next_q = self.Qnet.forward(new_state,new_depth,cs)
        idx=np.argmax(next_q.cpu().detach().numpy(),axis=1)
        
        temp=current_Q1
        target_action=T.ones(self.batch_size,1)
        for i in range(self.batch_size):
            target_action[i]=target_Q[i][idx[i]]
            TD_target = reward[i] + self.gamma*(target_action[i])*(1-done[i])

            temp[i][action[i]]=TD_target

        critic_loss = F.mse_loss(temp, current_Q1)

        # Optimize the critic
        self.Qnet.train()
        self.Qnet.optimizer.zero_grad()
        critic_loss.backward()
        self.Qnet.optimizer.step()
        actor_loss = 0

        # copy weights
        if self.count % self.update_freq == 0:

            self.update_network_parameters()

        self.count += 1

        return critic_loss.cpu().detach().numpy(), actor_loss


    def update_network_parameters(self):

        Qnet_params = self.Qnet.named_parameters()
        Qnet_dict = dict(Qnet_params)

        self.target_Qnet.load_state_dict(Qnet_dict)

    def save_models(self):

        self.Qnet.save_checkpoint()
        self.target_Qnet.save_checkpoint()

    def load_models(self):

        self.Qnet.load_checkpoint()
        self.target_Qnet.load_checkpoint()


class Eps:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_explor_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1 * current_step * self.decay)
