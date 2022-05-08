from .Speed_Buffer import ReplayBuffer
from .Mul_network import *


class mul_agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99,
                 n_actions=2, max_size=1000000, batch_size=64, token=0, update_freq=1, idx='0', eval=0, weight=0):
        # multi-objective weights
        weights = [[0.99, 0.01], [0.97, 0.03], [0.95, 0.05], [0.92, 0.08], [0.9, 0.1], [0.88, 0.12], [0.85, 0.15]]
        self.weight = weights[weight]

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.count = 0
        cuda_idx = 'cuda:' + idx
        self.device = T.device(cuda_idx if T.cuda.is_available() else 'cpu')
        chkpt_dir = './checkpoint/mul/' + str(token)
        self.noise_clip = 0.2
        self.mask = T.ones(self.batch_size).to(self.device)
        if eval == 0:
            os.mkdir(chkpt_dir)

        self.actor = ActorNetwork(alpha, input_dims, n_actions,
                                  'Actor', chkpt_dir=chkpt_dir, device=self.device)

        self.accuracy_critic = CriticNetwork(beta, input_dims, n_actions,
                                             'accuracy_critic', chkpt_dir=chkpt_dir, device=self.device)

        self.speed_critic = CriticNetwork(beta, input_dims, n_actions,
                                          'speed_critic', chkpt_dir=chkpt_dir, device=self.device)

        # **************************** target networks ************************************ #
        self.target_actor = ActorNetwork(alpha, input_dims, n_actions,
                                         'TargetActor', chkpt_dir=chkpt_dir, device=self.device)
        self.target_accuracy_critic = CriticNetwork(beta, input_dims, n_actions,
                                                    'Target_accuracy_critic', chkpt_dir=chkpt_dir, device=self.device)

        self.target_speed_critic = CriticNetwork(beta, input_dims, n_actions,
                                                 'Target_speed_critic', chkpt_dir=chkpt_dir, device=self.device)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, ratio):

        self.actor.eval()
        observation_scalar = observation[:-1]
        cross_section = observation[-1]
        observation = T.tensor(observation_scalar, dtype=T.float).to(self.device)
        cross_section = T.tensor(cross_section, dtype=T.float).to(self.device)
        mu = self.actor.forward(observation, cross_section, ratio).to(self.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.device).clamp(-self.noise_clip, self.noise_clip)
        self.actor.train()

        # make sure the noise doesn't make |mu| > 1
        temp = mu_prime.cpu().detach().numpy()
        actions = np.clip(temp, a_min=-1, a_max=1)

        # remap the value to be [-0.1,0.1]
        actions /= 10
        return actions

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self, ratio):

        if self.memory.mem_cntr < self.batch_size:
            return "nope", "too soon", "try again"

        state, action, reward, new_state, cross_section, new_cross_section, done = self.memory.sample(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        cross_section = T.tensor(cross_section, dtype=T.float).to(self.device)
        new_cross_section = T.tensor(new_cross_section, dtype=T.float).to(self.device)

        with T.no_grad():
            masked = T.sub(self.mask, done)
            masked = T.reshape(masked, (self.batch_size, 1))
            reward = T.reshape(reward, (self.batch_size, 1))

            # Select action according to policy and add clipped noise
            noise = (T.randn_like(action) * T.tensor(self.noise(), dtype=T.float).to(self.device)). \
                clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor.forward(new_state, cross_section, ratio) + noise).clamp(-1, 1) / 10

            # Compute the target Q value for accuracy objective
            target_accuracy_Q1, target_accuracy_Q2 = self.target_accuracy_critic(new_state, new_cross_section,
                                                                                 next_action)
            target_accuracy_Q = T.min(target_accuracy_Q1, target_accuracy_Q2)
            target_accuracy_Q = reward + masked * self.gamma * target_accuracy_Q

            # Compute the target Q value for speed objective
            target_speed_Q1, target_speed_Q2 = self.target_speed_critic(new_state, new_cross_section,
                                                                        next_action)
            target_speed_Q = T.min(target_speed_Q1, target_speed_Q2)
            target_speed_Q = reward + masked * self.gamma * target_speed_Q

        # Get current Q estimates for accuracy
        accuracy_current_Q1, accuracy_current_Q2 = self.accuracy_critic(state, cross_section, action)
        # Get current Q estimates for speed
        accuracy_speed_Q1, accuracy_speed_Q2 = self.speed_critic(state, cross_section, action)

        # Compute accuracy critic loss,
        accuracy_critic_loss = F.mse_loss(accuracy_current_Q1, target_accuracy_Q) \
                               + F.mse_loss(accuracy_current_Q2, target_accuracy_Q)

        # Compute speed critic loss
        speed_critic_loss = F.mse_loss(accuracy_speed_Q1, target_speed_Q) \
                            + F.mse_loss(accuracy_speed_Q2, target_speed_Q)

        # Optimize the critic
        self.accuracy_critic.train()
        self.accuracy_critic.optimizer.zero_grad()
        accuracy_critic_loss.backward()
        self.accuracy_critic.optimizer.step()

        self.speed_critic.train()
        self.speed_critic.optimizer.zero_grad()
        speed_critic_loss.backward()
        self.speed_critic.optimizer.step()

        actor_loss = 0
        # Delayed policy updates
        if self.count % self.update_freq == 0:
            # Only compute actor lose if both objectives are in the same direction
            accuracy_loss = T.mean(self.weight[0] * self.accuracy_critic.Q1(state, cross_section,
                                                                            self.actor.forward(state, cross_section,
                                                                                               ratio) / 10))
            speed_loss = T.mean(self.weight[1] * self.speed_critic.Q1(state, cross_section,
                                                                      self.actor.forward(state, cross_section,
                                                                                         ratio) / 10))
            if accuracy_loss > 0 and speed_loss > 0 or accuracy_loss < 0 and speed_loss < 0:
                actor_loss = T.mean(accuracy_loss + speed_loss)
                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                self.update_network_parameters()
                actor_loss = actor_loss.cpu().detach().numpy()
        self.count += 1
 
        return accuracy_critic_loss.cpu().detach().numpy(), speed_critic_loss.cpu().detach().numpy(), actor_loss

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        # soft copy from normal networks to target networks
        actor_params = self.actor.named_parameters()
        accuracy_critic_params = self.accuracy_critic.named_parameters()
        speed_critic_params = self.speed_critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_speed_critic_params = self.target_speed_critic.named_parameters()
        target_accuracy_critic_params = self.target_accuracy_critic.named_parameters()

        accuracy_critic_state_dict = dict(accuracy_critic_params)
        speed_critic_state_dict = dict(speed_critic_params)
        actor_state_dict = dict(actor_params)

        target_accuracy_critic_dict = dict(target_accuracy_critic_params)
        target_speed_critic_dict = dict(target_speed_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in speed_critic_state_dict:
            speed_critic_state_dict[name] = tau * speed_critic_state_dict[name].clone() + \
                                            (1 - tau) * target_speed_critic_dict[name].clone()
        for name in accuracy_critic_state_dict:
            accuracy_critic_state_dict[name] = tau * accuracy_critic_state_dict[name].clone() + \
                                               (1 - tau) * target_accuracy_critic_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_dict[name].clone()

        self.target_accuracy_critic.load_state_dict(accuracy_critic_state_dict)
        self.target_speed_critic.load_state_dict(speed_critic_state_dict)

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()

        self.accuracy_critic.save_checkpoint()
        self.target_accuracy_critic.save_checkpoint()

        self.speed_critic.save_checkpoint()
        self.target_speed_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()

        self.accuracy_critic.load_checkpoint()
        self.target_accuracy_critic.load_checkpoint()

        self.speed_critic.load_checkpoint()
        self.target_speed_critic.load_checkpoint()


# a special noise add to the action
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.1, theta=.2, dt=1e-2, x0=None):
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
 