

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
hyperparameters.neurons = [105, 256, 256, 8]
hyperparameters.lr = 0.0003
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
result_data = Results(EPISODES, y_range=6000)

# Replay
replay_memory = Memory(hyperparameters)

env = gym.make("Ant-v5", render_mode=RENDER_MODE)

try:
    steps_done = 0
    for episode in range(EPISODES):
        observation, info = env.reset()
        ongoing_reward = 0
        standard_deviations = []
        terminated = truncated = False 
        while not (terminated or truncated):
            result_data.update_image(env.render())
            with torch.no_grad():
                action, stdev = policy(observation)
                action = action.cpu().detach().numpy()
                stdev = stdev.cpu().detach().numpy()
            old_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)
            replay_memory.push(old_observation, action, observation, reward, terminated)
            optimizeSAC(replay_memory, hyperparameters, qs, qs_targets, policy, policy_optimizer, q_optimizers, log_of_alpha, alpha_optimizer)
            soft_weight_update(policy_target, policy, hyperparameters)
            for i in range(len(qs)):
                soft_weight_update(qs_targets[i], qs[i], hyperparameters)
            ongoing_reward += reward
            standard_deviations.append(stdev)
            steps_done += 1
        result_data.update_rewards(episode, ongoing_reward)
        result_data.update_decay(steps_done, np.mean(standard_deviations))
except KeyboardInterrupt:
    pass 

result_data.cleanup()


