from .PPO_network import *


class PPO_agent(object):
    def __init__(self, alpha, input_dims, gamma=0.99, ppo_update_eps=4,
                 n_actions=2, token=0, idx='0', eval=0, weight=0):
        # multi-objective weights
        weights = [[0.99, 0.01], [0.97, 0.03], [0.95, 0.05], [0.92, 0.08], [0.9, 0.1], [0.88, 0.12], [0.85, 0.15]]
        self.weight = weights[weight]
        self.GAE_LAMBDA = 0.95
        self.PPO_EPSILON = 0.2
        self.CRITIC_DISCOUNT = 0.5
        self.ENTROPY_BETA = 0.001
        self.MINI_BATCH_SIZE = 2

        # for each update, calculate the gradient PPO_EPOCHS times
        self.PPO_EPOCHS = ppo_update_eps

        self.gamma = gamma

        cuda_idx = 'cuda:' + idx
        self.device = T.device(cuda_idx if T.cuda.is_available() else 'cpu')
        chkpt_dir = './checkpoint/' + str(token)

        if eval == 0:
            os.mkdir(chkpt_dir)

        self.actor_critic = ActorCritic(alpha, input_dims, n_actions,
                                        'ActorCritic', chkpt_dir=chkpt_dir).to(self.device)

    def ppo_iter(self, batch_size, states, cross_sections, actions, log_probs, returns, advantages):
        batch_size = batch_size
        batch_start = np.arange(0, batch_size, self.MINI_BATCH_SIZE)
        np.random.shuffle(batch_start)

        for i in batch_start:
            yield states[i: i + self.MINI_BATCH_SIZE], cross_sections[i:i + self.MINI_BATCH_SIZE], \
                  actions[i:i + self.MINI_BATCH_SIZE], log_probs[i:i + self.MINI_BATCH_SIZE], \
                  returns[i:i + self.MINI_BATCH_SIZE], advantages[i:i + self.MINI_BATCH_SIZE]

            # using yield helps us to return the result multiple times instead of result one bulk
            # when batch 0 is ready, give it to the caller immediately, then come back here to prepare batch 2

    def ppo_update(self, num_step, states, cross_sections, actions, log_probs, returns, advantages, kappa,
                   clip_param=0.2):
        losses = []
        for _ in range(self.PPO_EPOCHS):
            for state, cross_section, action_tensor, old_log_prob_tensor, return_tensor, advantage_tensor \
                    in self.ppo_iter(num_step, states,
                                     cross_sections,
                                     actions,
                                     log_probs, returns,
                                     advantages):
                dist, value = self.actor_critic(state, cross_section, kappa)

                entropy = dist.entropy().mean()
                new_log_prob = dist.log_prob(action_tensor)

                # match the shape
                ratio = T.transpose((new_log_prob - old_log_prob_tensor).exp(), 0, 1)

                surr1 = ratio * advantage_tensor
                surr2 = T.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_tensor

                actor_loss = - T.min(surr1, surr2).mean()
                critic_loss = T.sqrt((return_tensor - value).pow(2).mean())

                loss = 0.5 * critic_loss + actor_loss - self.ENTROPY_BETA * entropy

                self.actor_critic.optimizer.zero_grad()
                loss.backward()
                self.actor_critic.optimizer.step()
                losses.append(loss.cpu().detach().numpy())
        return losses

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def save_models(self):

        self.actor_critic.save_checkpoint()

    def load_models(self):
        self.actor_critic.load_checkpoint()
