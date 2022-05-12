from ..replay_buffer.oneD_Buffer import oneD_ReplayBuffer as ReplayBuffer
from .network_torch import *


class DDPG_Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99,
                 n_actions=2, max_size=1000000, batch_size=64, token=0, update_freq=1, idx='0', eval=0):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.count = 0
        cuda_idx = 'cuda:' + idx
        self.device = T.device(cuda_idx if T.cuda.is_available() else 'cpu')
        chkpt_dir = './checkpoint/ddpg/' + str(token)
        self.mask = T.ones(self.batch_size).to(self.device)
        if eval == 0:
            os.mkdir(chkpt_dir)

        self.actor = ActorNetwork(alpha, input_dims, n_actions,
                                  'Actor', chkpt_dir=chkpt_dir, device=self.device)
        self.critic = CriticNetwork(beta, input_dims, n_actions,
                                    'Critic', chkpt_dir=chkpt_dir, device=self.device)

        self.target_actor = ActorNetwork(alpha, input_dims, n_actions,
                                         'TargetActor', chkpt_dir=chkpt_dir, device=self.device)
        self.target_critic = CriticNetwork(beta, input_dims, n_actions,
                                           'TargetCritic', chkpt_dir=chkpt_dir, device=self.device)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, ratio):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.device)

        mu = self.actor.forward(observation, ratio).to(self.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.device)
        self.actor.train()

        # make sure the noise doesn't make |mu| > 1
        temp = mu_prime.cpu().detach().numpy()
        actions = np.clip(temp, a_min=-1, a_max=1)

        # remap the value to be [-0.1,0.1]
        # actions /= 10
        return actions

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self, ratio):
        # only start training after we have enough samples 
        if self.memory.mem_cntr < self.batch_size:
            return "nope", "try again"

        state, action, reward, new_state, done = self.memory.sample(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # get s', a' using target networks 
        target_actions = self.target_actor.forward(new_state, ratio)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        # update the critic network 

        masked = T.sub(self.mask, done)
        masked = T.reshape(masked, (self.batch_size, 1))

        td_v = T.mul(critic_value_, masked)
        discounted_td_v = T.mul(td_v, self.gamma)

        reward = T.reshape(reward, (self.batch_size, 1))

        target = T.add(reward, discounted_td_v)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()

        # get the TD error
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # gradient ascend for actor 
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state, ratio)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        if self.count % self.update_freq == 0:
            self.update_network_parameters()
        self.count += 1

        return critic_loss.cpu().detach().numpy(), actor_loss.cpu().detach().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        # soft copy from normal networks to target networks
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()


# a special noise add to the action 
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)
