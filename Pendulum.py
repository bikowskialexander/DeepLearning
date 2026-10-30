


import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import math 
import time 

from Helpers.ControllerDDPG import ActorDDPG, CriticDDPG
from Helpers.ControllerDDPG import get_action_array, optimize, hard_weight_update, soft_weight_update, get_eps

from Helpers.Memory import Memory
from Helpers.Results import Results
from Helpers.Hyperparameters import HyperparameterData


EPISODES = 1000
RENDER_MODE = "rgb_array"


hyperparameters = HyperparameterData()
hyperparameters.lr = 0.0003
hyperparameters.tau = 0.005
hyperparameters.batch_size = 128
hyperparameters.eps_decay = 100000
hyperparameters.eps_start = 0.99
hyperparameters.eps_end = 0.10
hyperparameters.gamma = 0.99
hyperparameters.buffer_size = 1000000
hyperparameters.neurons = [4, 128, 128, 1]
hyperparameters.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

replay_memory = Memory(hyperparameters)

tracked_results = Results(EPISODES, y_range=1000)

# Networks
policy = ActorDDPG(hyperparameters)
q_function = CriticDDPG(hyperparameters)
policy_target = ActorDDPG(hyperparameters)
q_function_target = CriticDDPG(hyperparameters)

# Optimizers
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=hyperparameters.lr)
q_optimizer = torch.optim.Adam(q_function.parameters(), lr=10*hyperparameters.lr)
hard_weight_update(policy_target, policy)
hard_weight_update(q_function_target, q_function)

env = gym.make("InvertedPendulum-v5", render_mode="")
terminated = truncated = True 
for _ in tqdm(range(5000)):
    if terminated or  truncated:
        observation, info = env.reset()
    old_observation = observation
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    replay_memory.push(old_observation, action, observation, reward, terminated)

steps_done = 0
env = gym.make("InvertedPendulum-v5", render_mode=RENDER_MODE)
random_ng = np.random.default_rng()
try:
    for episode in tqdm(range(EPISODES)):
        observation, info = env.reset()
        terminated = truncated = False 
        ongoing_reward = 0 
        ongoing_noise = 0
        while not (terminated or truncated):
            
            # Get Action
            with torch.no_grad():
                observation_tensor = torch.from_numpy(observation).to(hyperparameters.device)
                action_tensor = policy(observation_tensor.unsqueeze(0).float())
                action = action_tensor.cpu().numpy().reshape((-1,))
                random_noise = random_ng.normal(0, 0.05, size=(1))
                action += random_noise
                action = np.clip(action, -1.0, 1.0)

                # Save old Observation
                old_observation = observation
            
            # Take Env Step
            observation, reward, terminated, truncated, info = env.step(3*action)

            # Save to Replay Memory
            replay_memory.push(old_observation, action, observation, reward, terminated)
            
            # Optimize 
            optimize(replay_memory, hyperparameters, policy, policy_target, q_function, q_function_target, q_optimizer, policy_optimizer)
            
            # Update Target Weights
            soft_weight_update(policy_target, policy, hyperparameters)
            soft_weight_update(q_function_target, q_function, hyperparameters)
            
            # Update Steps and Reward
            steps_done += 1
            ongoing_reward += reward
            ongoing_noise += math.fabs(random_noise[0])

            # Update Image
            frame = env.render()
            tracked_results.update_image(frame)
        tracked_results.update_decay(steps_done, ongoing_noise)
        tracked_results.update_rewards(episode, ongoing_reward)

    with torch.no_grad():
        truncated = truncated = True
        while True:
            if terminated or truncated:
                observation, info = env.reset()
            tracked_results.update_image(env.render())
            observation_tensor = torch.from_numpy(observation).to(hyperparameters.device)
            action = policy(observation_tensor.unsqueeze(0).float())
            action = action.cpu().numpy().reshape((-1,))
            observation, reward, terminated, truncated, info = env.step(3*action)
            time.sleep(0.01)

except KeyboardInterrupt:
    pass 

    

tracked_results.save_model(policy, "Final.pt")
tracked_results.save_config(hyperparameters)
tracked_results.save_result_graph()
tracked_results.cleanup()


