from ..replay_buffer.oneD_Buffer import oneD_ReplayBuffer as ReplayBuffer
from .TD3_s_network import *


class TD3_agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99,
                 n_actions=2, max_size=1000000, batch_size=64, token=0, update_freq=1, idx='0', eval=0):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        layer1_size = 400
        layer2_size = 300
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.update_freq = update_freq

        cuda_idx = 'cuda:' + idx
        self.device = T.device(cuda_idx if T.cuda.is_available() else 'cpu')
        chkpt_dir = './checkpoint/td3/se_net/' + str(token)
        self.noise_clip = 0.004
        self.action_count = 0
        self.learning_count = 0
        self.warmup = 1000
        self.mask = T.ones(self.batch_size).to(self.device)
        self.eval = eval
        if eval == 0:
            os.mkdir(chkpt_dir)

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='actor', chkpt_dir=chkpt_dir, device=self.device)
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='target_actor', chkpt_dir=chkpt_dir, device=self.device)

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name='critic_1', chkpt_dir=chkpt_dir, device=self.device)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions=n_actions,
                                             name='target_critic_1', chkpt_dir=chkpt_dir, device=self.device)

        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name='critic_2', chkpt_dir=chkpt_dir, device=self.device)
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions=n_actions,
                                             name='target_critic_2', chkpt_dir=chkpt_dir, device=self.device)
        # self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.noise = 0.1
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, ratio):
        self.actor.eval()
        if self.action_count < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.device)
        else:
            observation1 = observation[:-1]
            cross_section = observation[-1]
            observation = T.tensor(observation1, dtype=T.float).to(self.device)
            cross_section = T.tensor(cross_section, dtype=T.float).to(self.device)
            state = T.tensor(observation, dtype=T.float).to(self.device)
            mu = self.actor.forward(state, cross_section, ratio).to(self.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                 dtype=T.float).to(self.device)
        mu_prime = T.clamp(mu_prime, -1, 1)
        self.action_count += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self, ratio):

        if self.memory.mem_cntr < self.batch_size:
            return "nope", "try again"

        state, action, reward, new_state, cross_section, new_cross_section, done = self.memory.sample(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        cross_section = T.tensor(cross_section, dtype=T.float).to(self.device)
        new_cross_section = T.tensor(new_cross_section, dtype=T.float).to(self.device)

        target_actions = self.target_actor.forward(new_state, new_cross_section, ratio)
        target_actions = target_actions + \
                         T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # might break if elements of min and max are not all equal
        target_actions = T.clamp(target_actions, -1, 1)

        q1_ = self.target_critic_1.forward(new_state, new_cross_section, target_actions)
        q2_ = self.target_critic_2.forward(new_state, new_cross_section, target_actions)

        q1 = self.critic_1.forward(state, cross_section, action)
        q2 = self.critic_2.forward(state, cross_section, action)

        masked = T.sub(self.mask, done)
        masked = T.reshape(masked, (self.batch_size, 1))
        reward = T.reshape(reward, (self.batch_size, 1))

        # q1_[done] = 0.0
        # q2_[done] = 0.0

        # q1_ = q1_.view(-1)
        # q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + masked * self.gamma * critic_value_
        # target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learning_count += 1
        actor_loss = 0

        if self.learning_count % self.update_freq != 0:
            return critic_loss.cpu().detach().numpy(), actor_loss
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, cross_section, self.actor.forward(state, cross_section, ratio))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
        return critic_loss.cpu().detach().numpy(), actor_loss.cpu().detach().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + \
                                        (1 - tau) * target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + \
                                        (1 - tau) * target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()