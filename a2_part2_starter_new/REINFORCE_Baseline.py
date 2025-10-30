from math import log
import gymnasium as gym # Changed from gym to gymnasium
import numpy as np
import utils.envs as envs # Changed to use new envs_gymnasium
import utils.seed, utils.buffers, utils.torch
import torch, random
from torch import nn
import copy
import tqdm
import matplotlib.pyplot as plt
import warnings
import argparse
import os

warnings.filterwarnings("ignore")

# REINFORCE with Baseline

parser = argparse.ArgumentParser()

#either:
# cartpole - default cartpole environment
# mountain_car - default mountain car environment
# mountain_car_mod - mountain car environment with modified reward
parser.add_argument('--mode', type=str, default="cartpole") 

args = parser.parse_args()

# Constants
SEED = 1
t = utils.torch.TorchHelper()
DEVICE = t.device

#for cartpole
if args.mode == "cartpole":
    OBS_N = 4               # State space size
    ACT_N = 2               # Action space size
    ENV_NAME = "CartPole-v0"
    GAMMA = 1.0             # Discount factor in episodic reward objective
    LEARNING_RATE = 5e-4    # Learning rate for actor optimizer
elif "mountain_car" in args.mode:
    OBS_N = 2
    ACT_N = 3
    ENV_NAME = "MountainCar-v0"
    GAMMA = 0.9             # Discount factor in episodic reward objective
    LEARNING_RATE = 1e-3    # Learning rate for actor optimizer

EPOCHS = 800           # Total number of iterations to learn over
EPISODES_PER_EPOCH = 1  # Episodes per epoch (number of episodes observed, and batched)
TEST_EPISODES = 10      # Test episodes
HIDDEN = 32             # Hidden size
POLICY_TRAIN_ITERS = 10  # Number of iterations of policy improvement in each epoch


# Create environment
utils.seed.seed(SEED)
env = gym.make(ENV_NAME, render_mode="rgb_array") # Added render_mode and seed
# env.seed(SEED) # Removed as seed is passed to gym.make

# Networks
pi = torch.nn.Sequential(
    torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, ACT_N)
).to(DEVICE)

# Value network (Critic)
V = torch.nn.Sequential(
    torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, 1) # Output is a single value for V(s)
).to(DEVICE)

# Optimizers
OPT = torch.optim.Adam(pi.parameters(), lr = LEARNING_RATE)
V_OPT = torch.optim.Adam(V.parameters(), lr = LEARNING_RATE) # Using same LR for now

# Policy
def policy(env, obs):
    # Check if obs is a tuple (new gymnasium observation format)
    if isinstance(obs, tuple):
        obs = obs[0] # Extract the observation array from the tuple
    # Check if obs is a dictionary (if the observation itself is a dict)
    if isinstance(obs, dict) and 'observation' in obs:
        obs = obs['observation']
    probs = torch.nn.Softmax(dim=-1)(pi(t.f(obs)))
    return np.random.choice(ACT_N, p = probs.cpu().detach().numpy())

# Training function
# S = tensor of states observed in the episode/ batch of episodes
# A = tensor of actions taken in episode/ batch of episodes
# returns = tensor where nth element is \sum^{T-n}_0 gamma^n * reward (return at step n of episode)
def train(S,A,returns):

    #number of policy improvement steps given the observation
    for i in range(POLICY_TRAIN_ITERS):
        OPT.zero_grad()
        V_OPT.zero_grad()
        
        # Calculate log probabilities for policy update
        log_probs = torch.nn.LogSoftmax(dim=-1)(pi(S)).gather(1, A.view(-1, 1)).view(-1)
        
        # Calculate state values from the critic network
        state_values = V(S).squeeze() # V(s_t)
        
        # Calculate advantage A_t = G_t - V(s_t)
        advantages = returns - state_values.detach() # Detach to prevent gradients from flowing into V for policy update
        
        n = torch.arange(S.size(0)).to(DEVICE)
        
        # Policy loss (actor loss)
        # Objective: -((GAMMA**n) * advantages * log_probs).sum()
        policy_loss = -((GAMMA**n) * advantages * log_probs).sum()
        
        # Value loss (critic loss) - Mean Squared Error between G_t and V(s_t) (returns)
        value_loss = torch.nn.functional.mse_loss(state_values, returns)
        
        # Backpropagate and update actor
        policy_loss.backward()
        OPT.step()

        # Backpropagate and update critic
        value_loss.backward()
        V_OPT.step()

# Play episodes
Rs = [] 
last25Rs = []
print("Training:")
pbar = tqdm.trange(EPOCHS)
for epi in pbar:

    all_S, all_A = [], []
    all_returns = []

    # For this assignment we do 1 episode per epoch
    # This is if we collect multiple episodes, and perform an update on that batch
    # (the average gradient is applied)
    # this could result in smoother updates
    for epj in range(EPISODES_PER_EPOCH):
        
        # Play an episode and log episodic reward
        S, A, R = envs.play_episode(env, policy)

        #modify the reward for "mountain_car_mod" mode
        # replace reward with the height of the car (which is first component of state)
        if args.mode == "mountain_car_mod":
            R = [s[0] for s in S[:-1]]


        all_S += S[:-1] # ignore last state
        all_A += A
        
        # Create returns 
        discounted_rewards = copy.deepcopy(R)
        for i in range(len(R)-1)[::-1]:
            discounted_rewards[i] += GAMMA * discounted_rewards[i+1]
        discounted_rewards = t.f(discounted_rewards)
        all_returns += [discounted_rewards]

    Rs += [sum(R)]
    S, A = t.f(np.array(all_S)), t.l(np.array(all_A))
    returns = torch.cat(all_returns, dim=0).flatten()

    # train
    train(S, A, returns)

    # Show mean episodic reward over last 25 episodes
    last25Rs += [sum(Rs[-25:])/len(Rs[-25:])]
    pbar.set_description("R25(%g, mean over 10 episodes)" % (last25Rs[-1]))
  
pbar.close()
print("Training finished!")

# Plot the reward
N = len(last25Rs)
plt.plot(range(N), last25Rs, 'b')
plt.xlabel('Episode')
plt.ylabel('Reward (averaged over last 25 episodes)')
plt.title("REINFORCE with Baseline, mode: " + args.mode)

image_dir = "images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
plt.savefig(os.path.join(image_dir, "reinforce-baseline-"+args.mode+".png"))
print("Episodic reward plot saved!")

# Play test episodes
print("Testing:")
testRs = []
for epi in range(TEST_EPISODES):
    S, A, R = envs.play_episode(env, policy, render = False)

    #modify the reward for "mountain_car_mod" mode
    # replace reward with the height of the car (which is first component of state)
    if "mountain_car" in args.mode:
        R = [s[0] for s in S[:-1]]

    testRs += [sum(R)]
    print("Episode%02d: R = %g" % (epi+1, sum(R)))

if "mountain_car" in args.mode:
    print("Height achieved: %.2f ± %.2f" % (np.mean(testRs), np.std(testRs)))
else:
    print("Eval score: %.2f ± %.2f" % (np.mean(testRs), np.std(testRs)))

env.close()