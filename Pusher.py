
# =====================================================
# 
# Pusher.py
# Trains a actor to push the object into the circle 
# using a robotic arm form MuJoCo and gym.
# Soft Actor Critic is used for training.
# 
# =====================================================

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import math 

from Helpers.ControllerSAC import SoftQ_SAC, PolicySAC
from Helpers.Results import Results
from Helpers.Memory import Memory 
from Helpers.Memory import HyperparameterData

from Helpers.ControllerSAC import soft_weight_update, hard_weight_update, optimizeSAC, get_action_array

EPISODES = 4000
RENDER_MODE = "rgb_array"

# Hyperparameters 
hyperparameters = HyperparameterData()
hyperparameters.neurons = [23, 256, 256, 7]
hyperparameters.batch_size = 256
hyperparameters.lr = 0.0003
hyperparameters.tau = 0.005 
hyperparameters.gamma = 0.99
hyperparameters.device = device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Setup Memory
replay_memory = Memory(hyperparameters)

# Set up Nets
actor = PolicySAC(hyperparameters)
actor_target = PolicySAC(hyperparameters)
critic = SoftQ_SAC(hyperparameters)
critic_target = SoftQ_SAC(hyperparameters)

# Setup Optimization
actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=hyperparameters.lr)
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=hyperparameters.lr)
hard_weight_update(actor_target, actor)
hard_weight_update(critic_target, critic)

# Store Results
stored_results = Results(EPISODES, y_range=400)

env = gym.make('Pusher-v5', render_mode="")

terminated = truncated = False
for _ in tqdm(range(10000)):
    if (terminated or truncated):
        observation, info = env.reset()
        action = env.action_space.sample()
        old_state = observation
        observation, reward, terminated, truncated, info = env.step(action)
        replay_memory.push(old_state, action/2, observation, reward, terminated)

env = gym.make('Pusher-v5', render_mode=RENDER_MODE)

log_of_alpha = torch.zeros(1, requires_grad=True, device=hyperparameters.device)
alpha_optimizer = torch.optim.AdamW([log_of_alpha], lr=hyperparameters.lr)

try:
    step_count = 0
    for episode in tqdm(range(EPISODES)):
        observation, info = env.reset()
        terminated = truncated = False
        ongoing_reward = 0 
        stdevs = []
        while not (terminated or truncated): 
            observation = observation.astype(np.float32)
            action, std = get_action_array(actor, hyperparameters, observation)
            action *= 2
            old_state = observation
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            ongoing_reward += reward
            stdevs.append(std)
            replay_memory.push(old_state, action/2, observation, reward, terminated)

            # Optimize 
            optimizeSAC(replay_memory, hyperparameters, [critic], [critic_target], actor, actor_optimizer, [critic_optimizer], log_of_alpha, alpha_optimizer)

            # Soft Weight Update
            soft_weight_update(actor_target, actor, hyperparameters)
            soft_weight_update(critic_target, critic, hyperparameters)

            # Send Image
            stored_results.update_image(env.render())

        # Store Results
        stored_results.update_decay(step_count, np.mean(stdevs))
        stored_results.update_rewards(episode, ongoing_reward)
except KeyboardInterrupt:
    pass 

stored_results.save_model(actor,"Final.pt")
stored_results.save_config(hyperparameters)
stored_results.save_result_graph()
stored_results.cleanup()


