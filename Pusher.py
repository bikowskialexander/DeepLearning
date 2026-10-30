
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

from Helpers.ControllerSAC import SoftQ_SAC, PolicySAC
from Helpers.Results import Results
from Helpers.Memory import Memory 
from Helpers.Memory import HyperparameterData

from Helpers.ControllerSAC import soft_weight_update, hard_weight_update, optimizeSAC, get_action_array

# Episode Count
EPISODES = 2000

# Render Mode
# rgb_array should be used if also using live graphing.
# The display will be loaded on a different process
# through the Results class.
RENDER_MODE = "rgb_array"

# Hyperparameters 
hyperparameters = HyperparameterData()
hyperparameters.neurons = [23, 128, 128, 7]
hyperparameters.batch_size = 256
hyperparameters.lr = 0.0003
hyperparameters.tau = 0.005 
hyperparameters.gamma = 0.99
hyperparameters.device = device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Setup Memory
replay_memory = Memory(hyperparameters)

# Set up Nets
# Only one Q network used for pusher
actor = PolicySAC(hyperparameters)
actor_target = PolicySAC(hyperparameters)
critic = SoftQ_SAC(hyperparameters)
critic_target = SoftQ_SAC(hyperparameters)

# Set weights of targets to match actual 
hard_weight_update(actor_target, actor)
hard_weight_update(critic_target, critic)

# Setup Optimization
actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=hyperparameters.lr)
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=hyperparameters.lr)

# Store Results
stored_results = Results(EPISODES, y_range=200)

# =====================================================
#       Fill the replay with some random examples
# =====================================================

# Set Up Env
# Render mode should not be seen since
# all moves are random.
env = gym.make('Pusher-v5', render_mode="")

# Fill replay loop
terminated = truncated = False
for _ in tqdm(range(10000)):
    if (terminated or truncated):
        observation, info = env.reset()
        action = env.action_space.sample()
        old_state = observation
        observation, reward, terminated, truncated, info = env.step(action)
        replay_memory.push(old_state, action/2, observation, reward, terminated)

# =====================================================
#                    Do Training
# =====================================================

# Set Up env with render mode
env = gym.make('Pusher-v5', render_mode=RENDER_MODE)

# Set up alpha used for optimization 
log_of_alpha = torch.zeros(1, requires_grad=True, device=hyperparameters.device)
alpha_optimizer = torch.optim.AdamW([log_of_alpha], lr=hyperparameters.lr)

# Main Loop
try:
    step_count = 0
    for episode in tqdm(range(EPISODES)):

        # Set up for new run
        observation, info = env.reset()
        terminated = truncated = False
        ongoing_reward = 0 
        stdevs = []

        # Episode Loop
        while not (terminated or truncated): 

            # Ensure observation is right type for env
            observation = observation.astype(np.float32)

            # Get action form policy network
            action, std = get_action_array(actor, hyperparameters, observation)
            
            # Scale action to match env
            # The network only outputs from -1 to 1
            # but the env can take inputs from -2 to 2
            action *= 2

            # Save old state for replay buffer
            old_state = observation

            # Take step in env
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Update steps, reward, and standard deviation logs
            step_count += 1
            ongoing_reward += reward
            stdevs.append(std)
            
            # Add to replay memory
            replay_memory.push(old_state, action/2, observation, reward, terminated)

            # Optimize using soft actor critic
            optimizeSAC(replay_memory, hyperparameters, [critic], [critic_target], actor, actor_optimizer, [critic_optimizer], log_of_alpha, alpha_optimizer)

            # Soft Weight Update
            soft_weight_update(actor_target, actor, hyperparameters)
            soft_weight_update(critic_target, critic, hyperparameters)

            # Send Image
            stored_results.update_image(env.render())

        # Store Results
        stored_results.update_decay(step_count, np.mean(stdevs))
        stored_results.update_rewards(step_count, ongoing_reward)

except KeyboardInterrupt:
    pass 


# =====================================================
#               Save Results and Cleanup
# =====================================================

stored_results.save_model(actor,"Final.pt")
stored_results.save_config(hyperparameters)
stored_results.save_result_graph()
stored_results.cleanup()


