import time
import argparse as ap
from environment import coppelia
import matplotlib.pyplot as plt

# ************** DDPG *******************
from algorithms.DDPG.ddpg_torch import *
from algorithms.DDPG.vision_ddpg_torch import Vision_Agent
from algorithms.DDPG.s_ddpg_torch import s_Agent

# ************** TD3 *******************
from algorithms.TD3.vision_td3_torch import Vision_td3_Agent
from algorithms.TD3.TD3_agent import TD3_agent
from algorithms.TD3.TD3_speed import TD3_speed_agent
from algorithms.mul_obj_TD3.Mul_agent import mul_agent

# ************** DDQN *******************
from algorithms.DDQN.DQN_agent import DQN_agent
from algorithms.DDQN.vision_DQN_agent import Vision_DQN_agent
from algorithms.DDQN.DQN_s_agent import DQN_s_agent

# ************** PPO *******************
from algorithms.PPO.PPO_agent import PPO_agent

parser = ap.ArgumentParser()
parser.add_argument("port", help="choose a port",
                    choices={'19990', '19991', '19992', '19993', '19994', '19995', '19996', '19997', '19998', '19999'})

# parser.add_argument("alpha", help="choose the learning rate for actor",
#                     choices={'0.00001', '0.000025', '0.00005','0.0001'})
#
# parser.add_argument("beta", help="choose the learning rate for critic",
#                     choices={'0.0001', '0.00025', '0.0005','0.001'})
parser.add_argument('discount', help="gamma", choices={'0', '0.5', '0.99', '1'})

parser.add_argument('update_freq', help="how often we update target networks",
                    choices={'1', '2', '4', '5', '6', '7', '8'})
parser.add_argument('cuda', help="use which gpu", choices={'0', '1'})

parser.add_argument('depth', help="use vision or not", choices={'0', '1'})
parser.add_argument('model', help='DDPG,TD3,DQN', choices={'0', '1', '2', '3', '4', '5', '6', '7'})
parser.add_argument('weight', help="objective weights", choices={'0', '1', '2', '3', '4', '5', '6', '7'})
parser.add_argument('eval', help="eval or train", choices={'0', '1'})
# parser.add_argument('same', help="same or eval setting", choices={'2', '1'})

args = parser.parse_args()

# alpha = float(args.alpha)
# beta = float(args.beta)
# batch_size=int(args.batch)
# network_type=int(args.cnn)
alpha = 0.000025
beta = 0.00025
batch_size = 64
network_type = 0
gamma = float(args.discount)
depth = int(args.depth)
port = int(args.port)
token = str(round(time.time()))
update_freq = int(args.update_freq)
eval = bool(int(args.eval))
idx = args.cuda
model = int(args.model)
weight = int(args.weight)
# same = args.same

if model == 0:
    # token='1642562823'

    if depth == 0:
        agent = DDPG_Agent(alpha=alpha, beta=beta, input_dims=[4], tau=0.001, gamma=gamma,
                      batch_size=batch_size, n_actions=2, token=token, update_freq=update_freq, idx=idx, eval=eval)
    if depth == 1:
        agent = Vision_Agent(alpha=alpha, beta=beta, input_dims=[4], tau=0.001, gamma=gamma,
                             batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx,
                             eval=eval)
elif model == 1:
    # token='1642267600'
    if depth == 0:
        agent = TD3_agent(alpha=alpha, beta=beta, input_dims=[4], tau=0.001, gamma=gamma,
                          batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx, eval=eval)
    if depth == 1:
        agent = Vision_td3_Agent(alpha=alpha, beta=beta, input_dims=[4], tau=0.001, gamma=gamma,
                                 batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx,
                                 eval=eval)

elif model == 2:
    # token='1642625149'

    if depth == 0:
        agent = DQN_agent(alpha=alpha, beta=beta, input_dims=[7], tau=0.001, gamma=gamma,
                          batch_size=batch_size, n_actions=9, token=token, update_freq=update_freq, idx=idx, eval=eval)
    if depth == 1:
        agent = Vision_DQN_agent(alpha=alpha, beta=beta, input_dims=[4], tau=0.001, gamma=gamma,
                                 batch_size=batch_size, n_actions=27, token=token, update_freq=update_freq, idx=idx,
                                 eval=eval)
elif model == 3:
    token='1649383425'

    agent = TD3_speed_agent(alpha=alpha, beta=beta, input_dims=[5], tau=0.001, gamma=gamma,
                            batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx,
                            eval=eval)
elif model == 4:
    agent = s_Agent(alpha=alpha, beta=beta, input_dims=[4], tau=0.001, gamma=gamma,
                    batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx,
                    eval=eval)

elif model == 5:
    agent = mul_agent(alpha=alpha, beta=beta, input_dims=[5], tau=0.001, gamma=gamma,
                      batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx, weight=weight,
                      eval=eval)
elif model == 6:
    # # no buffer, so input = 6 to include visual input
    agent = PPO_agent(alpha=alpha, input_dims=[5], gamma=gamma, n_actions=3, token=token,
                      ppo_update_eps=update_freq, idx=idx, weight=weight, eval=eval)
    # agent = Agent2(3,5)
else:
    agent = DQN_s_agent(alpha=alpha, beta=beta, input_dims=[4], tau=0.001, gamma=gamma,
                        batch_size=batch_size, n_actions=27, token=token, update_freq=update_freq, idx=idx, eval=eval)

# start coppelia connection
env = coppelia(depth, token, port=port, model=model, eval=eval, )
# same=same)

TD_error = []
score_history = []
action_history = []
actor_loss_history = []
critic_loss_history = []
outlier_history = []
patient = 200
warm_up = 20
best_score = -1000
best_episode = 90

# surface to volume ratio
s_to_v = np.array([[240, 315.8, 500]  # cuboid
                      , [240, 340.56, 375]  # cylinder
                      , [240, 352.94, 375]  # sphere
                   ]) / 500

cuda_idx = 'cuda:' + idx

device = T.device(cuda_idx if T.cuda.is_available() else 'cpu')
if eval:
    agent.load_models()
    np.random.seed()
    # token += same
else:
    np.random.seed(0)

# when using PPO, apply a new traning loop
if model == 6:
    num_step = int(input("Please enter number of step for each update:\n"))

    for i in range(1000):

        # random initial amount and shape/size
        mypot = np.random.randint(5)
        obj_shape = np.random.randint(3)
        size = np.random.randint(3)

        if depth:
            size = 0
        else:
            obj_shape = 0

        # value to indicate size/shape
        ratio = [s_to_v[obj_shape][size]]
        ratio = T.tensor(ratio, dtype=T.float).to(device)

        obs = env.reset(mypot, obj_shape, size)

        done = False
        score = 0
        # ii=0
        while not done:

            log_probs = []
            values = []
            states = []
            cross_sections = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            # within num_step, not updating the network
            # only using it to produce trajectory
            for _ in range(num_step):
                if done:
                    break
                current_state = obs[:-1]
                cross_section = obs[-1]
                current_state = T.tensor(current_state, dtype=T.float).to(device)
                cross_section = T.tensor(cross_section, dtype=T.float).to(device)
                dist, value = agent.actor_critic(current_state, cross_section, ratio)

                action = dist.sample()
                clipped_action = action.cpu().numpy()

                scale = max(np.absolute(clipped_action)) / 0.1

                clipped_action /= scale
                clipped_action = np.clip(clipped_action, a_min=-0.1, a_max=0.1)/2

                next_state, reward, done = env.step(clipped_action,i)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                log_probs.append(log_prob.cpu().detach().numpy())
                values.append(value)
                rewards.append(reward)
                masks.append(1 - done)
                states.append(obs[:-1])
                cross_sections.append(obs[-1])
                actions.append(clipped_action)
                action_history.append(clipped_action)
                score += reward
                obs = next_state

            cross_section = next_state[-1]
            next_state = next_state[:-1]
            cross_section = T.tensor(cross_section, dtype=T.float).to(device)
            next_state = T.tensor(next_state, dtype=T.float).to(device)
            _, next_value = agent.actor_critic(next_state, cross_section, ratio)

            returns = agent.compute_gae(next_value, rewards, masks, values)

            returns = T.cat(returns).detach()
            values = T.cat(values).detach()
            log_probs = np.array(log_probs)
            states = np.array(states)
            actions = np.array(actions)
            cross_sections = np.array(cross_sections)
            log_probs = T.tensor(log_probs, dtype=T.float).to(agent.device)
            states = T.tensor(states, dtype=T.float).to(agent.device)
            cross_sections = T.tensor(cross_sections, dtype=T.float).to(agent.device)
            actions = T.tensor(actions, dtype=T.float).to(agent.device)
            advantage = returns - values
            losses = agent.ppo_update(num_step, states, cross_sections, actions,
                             log_probs, returns, advantage, ratio)

            critic_loss_history.append(losses)
            # print(len(critic_loss_history), ii)
            # ii += 1
        num_outlier = env.finish()
        print(f'num_outlier {num_outlier}')
        outlier_history.append(num_outlier)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score and i > warm_up:
            best_score = avg_score
            if not eval:
                print('... saving checkpoint ...')
                agent.save_models()
            print("episode ", i, "score %.1f " % score, "best avg score %.1f " % avg_score)

            best_episode = i

        if i - best_episode > patient:
            print('early stop kicks in')
            break

        if i > 100:
            print('episode ', i, 'score %.2f' % score,
                  'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        print("-------------------")
else:
    for i in range(1000):
        # random initial amount and shape/size
        mypot = np.random.randint(5)
        obj_shape = np.random.randint(3)
        size = np.random.randint(3)

        if depth:
            size = 0
        else:
            obj_shape = 0

        # value to indicate size/shape
        ratio = [s_to_v[obj_shape][size]]
        ratio = T.tensor(ratio, dtype=T.float).to(device)

        obs = env.reset(mypot, obj_shape, size)
        if eval:
            agent.noise.reset()
        done = False
        score = 0
        while not done:

            if not depth:
                act = agent.choose_action(obs, ratio)
            else:
                act = agent.choose_action(obs)

            # grab the tuple
            new_state, reward, done = env.step(act)
            if not eval:
                agent.remember(obs, act, reward, new_state, int(done))
                if not depth:
                    critic_loss, actor_loss = agent.learn(ratio)
                else:
                    critic_loss, actor_loss = agent.learn()

                if not isinstance(actor_loss, str):
                    actor_loss_history.append(actor_loss)
                    critic_loss_history.append(critic_loss)
            score += reward
            obs = new_state

        num_outlier = env.finish()
        print(f'num_outlier {num_outlier}')
        outlier_history.append(num_outlier)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score and i > warm_up:
            best_score = avg_score
            if not eval:
                print('... saving checkpoint ...')
                agent.save_models()
            print("episode ", i, "score %.1f " % score, "best avg score %.1f " % avg_score)

            best_episode = i

        if i - best_episode > patient:
            print('early stop kicks in')
            break

        if i > 100:
            print('episode ', i, 'score %.2f' % score,
                  'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        print("-------------------")



# logging
reward_history = env.get_reward_history()
init_history, init_error = env.get_init_history()
print(f'token  {token}, init_error{init_error}')

dir_path = './npy/' + token
os.mkdir(dir_path)
path1 = os.path.join(dir_path, 'score.npy')
path2 = os.path.join(dir_path, 'avg.npy')
path3 = os.path.join(dir_path, 'critic.npy')
path4 = os.path.join(dir_path, 'actor.npy')
path5 = os.path.join(dir_path, 'out.npy')
path6 = os.path.join(dir_path, 'reward.npy')

running_avg = np.zeros(len(score_history))
for i in range(len(running_avg)):
    running_avg[i] = np.mean(score_history[max(0, i - 100):(i + 1)])

with open(path1, 'wb') as f:
    np.save(f, np.array(score_history))
with open(path2, 'wb') as f:
    np.save(f, np.array(running_avg))

if not eval and model != 6:
    with open(path3, 'wb') as f:
        np.save(f, np.array(critic_loss_history))
    pl = np.zeros((1, 3))
    # action_history = np.array(action_history)
    # for i in range(len(action_history)):
        # pl = np.concatenate((pl, [action_history[i]]))
    with open(path4, 'wb') as f:
        np.save(f, np.array(action_history))

elif model == 6:
    with open(path3, 'wb') as f:
        np.save(f, np.array(critic_loss_history).flatten())
    with open(path4, 'wb') as f:
        np.save(f, np.array(actor_loss_history))

with open(path5, 'wb') as f:
    np.save(f, np.array(outlier_history))

with open(path6, 'wb') as f:
    np.save(f, np.array(reward_history))

print(init_history,num_step)
