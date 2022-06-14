import time
import argparse as ap
from environment_classic import classic_coppelia
from environment_sin import sin_coppelia
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

parser.add_argument('discount', help="gamma", choices={'0', '0.5', '0.99', '1'})

parser.add_argument('update_freq', help="how often we update target networks",
                    choices={'1', '2', '4', '5', '6', '7', '8'})
parser.add_argument('cuda', help="use which gpu", choices={'0', '1'})

parser.add_argument('depth', help="use vision or not", choices={'0', '1'})
parser.add_argument('model', help='DDPG,TD3,DQN', choices={'0', '1', '2', '3', '4', '5', '6', '7'})
parser.add_argument('eval', help="eval or train", choices={'0', '1'})
parser.add_argument('sin', help="use sin or not", choices={'0', '1'})
args = parser.parse_args()

# alpha = float(args.alpha)
# beta = float(args.beta)
# batch_size=int(args.batch)

alpha = 0.0001
beta = 0.0001
batch_size = 100
tau = 0.005
gamma = float(args.discount)
depth = int(args.depth)
port = int(args.port)
token = str(round(time.time()))
update_freq = int(args.update_freq)
eval = bool(int(args.eval))
idx = args.cuda
model = int(args.model)
sin = args.sin
# same = args.same

if model == 0:
    token = '1652'
    if depth == 0:
        agent = DDPG_Agent(alpha=alpha, beta=beta, input_dims=[5], tau=tau, gamma=gamma,
                           batch_size=batch_size, n_actions=1, token=token, update_freq=update_freq, idx=idx, eval=eval)
    if depth == 1:
        agent = Vision_Agent(alpha=alpha, beta=beta, input_dims=[4], tau=tau, gamma=gamma,
                             batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx,
                             eval=eval)
elif model == 1:
    #token = '1655215430'
    if depth == 0:
        agent = TD3_agent(alpha=alpha, beta=beta, input_dims=[5], tau=tau, gamma=gamma,
                          batch_size=batch_size, n_actions=1, token=token, update_freq=update_freq, idx=idx, eval=eval)
    if depth == 1:
        agent = Vision_td3_Agent(alpha=alpha, beta=beta, input_dims=[4], tau=tau, gamma=gamma,
                                 batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx,
                                 eval=eval)

elif model == 2:
    # token='1642625149'

    if depth == 0:
        agent = DQN_agent(alpha=alpha, beta=beta, input_dims=[5], tau=tau, gamma=gamma,
                          batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx, eval=eval)
    if depth == 1:
        agent = Vision_DQN_agent(alpha=alpha, beta=beta, input_dims=[4], tau=tau, gamma=gamma,
                                 batch_size=batch_size, n_actions=27, token=token, update_freq=update_freq, idx=idx,
                                 eval=eval)
elif model == 3:
    # token = '1649383425'

    agent = TD3_speed_agent(alpha=alpha, beta=beta, input_dims=[5], tau=tau, gamma=gamma,
                            batch_size=batch_size, n_actions=2, token=token, update_freq=update_freq, idx=idx,
                            eval=eval)
elif model == 4:
    agent = s_Agent(alpha=alpha, beta=beta, input_dims=[5], tau=tau, gamma=gamma,
                    batch_size=batch_size, n_actions=2, token=token, update_freq=update_freq, idx=idx,
                    eval=eval)

elif model == 5:
    agent = mul_agent(alpha=alpha, beta=beta, input_dims=[5], tau=tau, gamma=gamma,
                      batch_size=batch_size, n_actions=3, token=token, update_freq=update_freq, idx=idx,
                      eval=eval)
elif model == 6:
    # # no buffer, so input = 6 to include visual input
    agent = PPO_agent(alpha=alpha, input_dims=[5], gamma=gamma, n_actions=3, token=token,
                      ppo_update_eps=update_freq, idx=idx, eval=eval)
    # agent = Agent2(3,5)
else:
    agent = DQN_s_agent(alpha=alpha, beta=beta, input_dims=[5], tau=tau, gamma=gamma,
                        batch_size=batch_size, n_actions=9, token=token, update_freq=update_freq, idx=idx, eval=eval)

# start coppelia connection
if sin == '0':
    env = classic_coppelia(depth, token, port=port, model=model, eval=eval, )
else:
    print('using sin wave')
    env = sin_coppelia(depth, token, port=port, model=model, eval=eval, )
# same=same)

TD_error = []
score_history = []
action_history = []
actor_loss_history = []
critic_loss_history = []
outlier_history = []
pour_out_bins = []
patient = 300
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

    print(f'testing with {token} in model {model} ')
    agent.load_models()
    # token += same

np.random.seed()

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

    #if eval:
    #    agent.noise.reset()

    done = False
    score = 0
    while not done:

        if not depth:
            act = agent.choose_action(obs, ratio)
        else:
            act = agent.choose_action(obs)

        # for the first 10 episodes, use the linear regression result
        # to help the help explore
        new_state, reward, done = env.step(act, i)
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

    num_outlier, num_poured_out = env.finish()
    print(f'num_outlier {num_outlier}, total poured out {num_poured_out}')
    if sin == '1':
        pour_out_bins.append(num_poured_out)
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

if not eval:
    with open(path3, 'wb') as f:
        np.save(f, np.array(critic_loss_history))
    pl = np.zeros((1, 3))

    with open(path4, 'wb') as f:
        np.save(f, np.array(actor_loss_history))

with open(path5, 'wb') as f:
    np.save(f, np.array(outlier_history))

with open(path6, 'wb') as f:
    np.save(f, np.array(reward_history))

if sin == '1':
    path7 = os.path.join(dir_path, 'poured.npy')
    with open(path7, 'wb') as f:
        np.save(f, np.array(pour_out_bins))
