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

# PPO

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
    EPOCHS = 150            # PPO typically converges faster
elif "mountain_car" in args.mode:
    OBS_N = 2
    ACT_N = 3
    ENV_NAME = "MountainCar-v0"
    GAMMA = 0.9             # Discount factor in episodic reward objective
    LEARNING_RATE = 1e-3    # Learning rate for actor optimizer
    EPOCHS = 150            # PPO typically converges faster

# EPOCHS = 800           # Total number of iterations to learn over (moved to env-specific)
EPISODES_PER_EPOCH = 1  # Episodes per epoch (number of episodes observed, and batched)
TEST_EPISODES = 10      # Test episodes
HIDDEN = 32             # Hidden size
POLICY_TRAIN_ITERS = 1  # Number of iterations of policy improvement in each epoch (for REINFORCE)

# PPO specific constants
CLIP_EPSILON = 0.2 # PPO clipping parameter
PPO_EPOCHS = 10    # Number of PPO optimization epochs over the collected batch


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
V_OPT = torch.optim.Adam(V.parameters(), lr = LEARNING_RATE)

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
# old_log_probs = log probabilities of actions taken under the old policy
def train(S,A,returns, old_log_probs):

    # PPO uses multiple epochs of optimization over the collected batch
    for _ in range(PPO_EPOCHS):
        OPT.zero_grad()
        V_OPT.zero_grad()
        
        # Calculate current policy's log probabilities
        current_log_probs = torch.nn.LogSoftmax(dim=-1)(pi(S)).gather(1, A.view(-1, 1)).view(-1)
        
        # Calculate state values from the critic network
        state_values = V(S).squeeze() # V(s_t)
        
        # Calculate advantage A_t = G_t - V(s_t)
        advantages = returns - state_values.detach() # Detach to prevent gradients from flowing into V for policy update
        
        # Calculate the ratio of new policy to old policy probabilities
        ratio = torch.exp(current_log_probs - old_log_probs.detach()) # Detach old_log_probs
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
        
        # Policy loss (actor loss) - PPO objective
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (critic loss) - Mean Squared Error between G_t and V(s_t)
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

    all_S, all_A, all_old_log_probs = [], [], []
    all_returns = []

    # For this assignment we do 1 episode per epoch
    # This is if we collect multiple episodes, and perform an update on that batch
    # (the average gradient is applied)
    # this could result in smoother updates
    for epj in range(EPISODES_PER_EPOCH):
        
        # Play an episode and log episodic reward
        S, A, R = envs.play_episode(env, policy)

        # Get log probabilities of actions taken under the current policy (which will be the old policy for the update)
        with torch.no_grad(): # No need to compute gradients for old_log_probs
            old_log_probs_episode = torch.nn.LogSoftmax(dim=-1)(pi(t.f(S[:-1]))).gather(1, t.l(np.array(A)).view(-1, 1)).view(-1)

        #modify the reward for "mountain_car_mod" mode
        # replace reward with the height of the car (which is first component of state)
        if args.mode == "mountain_car_mod":
            R = [s[0] for s in S[:-1]]

        all_S += S[:-1] # ignore last state
        all_A += A
        all_old_log_probs += old_log_probs_episode.cpu().numpy().tolist() # Store as list, convert to tensor later
        
        # Create returns 
        discounted_rewards = copy.deepcopy(R)
        for i in range(len(R)-1)[::-1]:
            discounted_rewards[i] += GAMMA * discounted_rewards[i+1]
        discounted_rewards = t.f(discounted_rewards)
        all_returns += [discounted_rewards]

    Rs += [sum(R)]
    S, A = t.f(np.array(all_S)), t.l(np.array(all_A))
    returns = torch.cat(all_returns, dim=0).flatten()
    old_log_probs = t.f(np.array(all_old_log_probs)).flatten() # Convert to tensor

    # train
    train(S, A, returns, old_log_probs) # Pass old_log_probs

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
plt.title("PPO, mode: " + args.mode)

image_dir = "images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
plt.savefig(os.path.join(image_dir, "ppo-"+args.mode+".png"))
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