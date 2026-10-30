# =====================================================
# 
# Ant.py
# Train policy for MuJoCo's Ant.
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
EPISODES = 5000

# Render Mode
# rgb_array should be used if also using live graphing.
# The display will be loaded on a different process
# through the Results class.
RENDER_MODE = "rgb_array"

# Hyperparameters 
hyperparameters = HyperparameterData()
hyperparameters.neurons = [105, 256, 256, 8]
hyperparameters.lr = 0.0003
hyperparameters.buffer_size = 10000000
hyperparameters.batch_size = 256
hyperparameters.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# Networks
policy = PolicySAC(hyperparameters)
qs = []
qs_targets = []
for q in range(2):
    qs.append(SoftQ_SAC(hyperparameters))
    qs_targets.append(SoftQ_SAC(hyperparameters))
policy_target = PolicySAC(hyperparameters)

# Set up alpha (automatic temperature)
log_of_alpha = torch.zeros(1, requires_grad=True, device=hyperparameters.device)

# Optimizers
policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=hyperparameters.lr)
q_optimizers = []
for i in range(2):
    q_optimizers.append(torch.optim.AdamW(qs[i].parameters(), lr=hyperparameters.lr)) 
alpha_optimizer = torch.optim.AdamW([log_of_alpha], lr=hyperparameters.lr)

# Results 
result_data = Results(EPISODES, y_range=5000)

# Replay
replay_memory = Memory(hyperparameters)

# Set up initial steps in env
env = gym.make("Ant-v5", render_mode="")
terminated = truncated = True
for _ in tqdm(range(10000)):
    if (terminated or truncated):
       observation, _ = env.reset()
    action = env.action_space.sample()
    old_observation = observation
    observation, reward, terminated, truncated, _ = env.step(action)
    replay_memory.push(old_observation, action, observation, reward, terminated)

# Set up env for training and rendering
env = gym.make("Ant-v5", render_mode=RENDER_MODE)

try:
    steps_done = 0
    for episode in range(EPISODES):
        observation, info = env.reset()
        ongoing_reward = 0
        standard_deviations = []
        terminated = truncated = False 

        # Episode Loop
        while not (terminated or truncated):

            # Update the image 
            result_data.update_image(env.render())

            # Get the action
            with torch.no_grad():
                action, stdev = policy(observation)
                action = action.cpu().detach().numpy()
                stdev = stdev.cpu().detach().numpy()

            # Save old state for recording in replay memory
            old_observation = observation

            # Take a step
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Push states, rewards, and actions to the replay memory 
            replay_memory.push(old_observation, action, observation, reward, terminated)

            # Optimize using Soft Actor Critic
            optimizeSAC(replay_memory, hyperparameters, qs, qs_targets, policy, policy_optimizer, q_optimizers, log_of_alpha, alpha_optimizer)

            # Update Weights            
            soft_weight_update(policy_target, policy, hyperparameters)
            for i in range(len(qs)):
                soft_weight_update(qs_targets[i], qs[i], hyperparameters)

            # Save data for graphing
            ongoing_reward += reward
            standard_deviations.append(stdev)
            steps_done += 1

        # Add data to graph
        result_data.update_rewards(episode, ongoing_reward)
        result_data.update_decay(steps_done, np.mean(standard_deviations))
        
except KeyboardInterrupt:
    pass 

# Save Data
result_data.save_model(policy, "Final.pt")
result_data.save_config(hyperparameters)
result_data.save_result_graph()
result_data.cleanup()


