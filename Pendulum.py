


import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import math 

from Helpers.ControllerDDPG import ActorDDPG, CriticDDPG
from Helpers.ControllerDDPG import get_action_array, optimize, hard_weight_update, soft_weight_update, get_eps

from Helpers.Memory import Memory
from Helpers.Results import Results
from Helpers.Hyperparameters import HyperparameterData


EPISODES = 500

def get_noise_scaler(reward, last_reward, lowest_reward):
    try:
        diff = math.fabs(reward - last_reward)*1000
        lowest_diff = math.fabs(lowest_reward - reward)*1000
        return np.clip( 0.5*(1/diff) + 2*(1/lowest_diff), 0.0, 1.0)
    except:
        return 1.0


def custom_reward(observation):
    reward = observation[0][0]
    if observation[0][0] > 0:
        reward += (1 - math.fabs(observation[1][0]))
    else:
        reward -= 2.0
    return reward


hyperparameters = HyperparameterData()
hyperparameters.lr = 0.0003
hyperparameters.tau = 0.005
hyperparameters.batch_size = 128
hyperparameters.eps_decay = 100000
hyperparameters.eps_start = 0.99
hyperparameters.eps_end = 0.10
hyperparameters.gamma = 0.80
hyperparameters.buffer_size = 1000000
hyperparameters.neurons = [3, 128, 128, 1]
hyperparameters.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

replay_memory = Memory(hyperparameters)

tracked_results = Results(EPISODES, y_range=600)

# Networks
policy = ActorDDPG(hyperparameters, result_multiplier=2.0)
q_function = CriticDDPG(hyperparameters)
policy_target = ActorDDPG(hyperparameters, result_multiplier=2.0)
q_function_target = CriticDDPG(hyperparameters)

# Optimizers
policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=hyperparameters.lr)
q_optimizer = torch.optim.AdamW(q_function.parameters(), lr=10*hyperparameters.lr)
hard_weight_update(policy_target, policy)
hard_weight_update(q_function_target, q_function)

env = gym.make("Pendulum-v1", render_mode="")
terminated = truncated = True 
for _ in tqdm(range(5000)):
    if terminated or  truncated:
        observation, info = env.reset()
    old_observation = observation.reshape((1, 3))
    action = env.action_space.sample().reshape((1,1))
    observation, reward, terminated, truncated, info = env.step(action)
    reward = custom_reward(observation)
    replay_memory.push(old_observation, action, observation.reshape((1, 3)), reward, terminated)

steps_done = 0
env = gym.make("Pendulum-v1", render_mode="")
lowest_reward = 99999999
rng = np.random.default_rng()
try:
    for episode in tqdm(range(EPISODES)):
        observation, info = env.reset()
        terminated = truncated = False 
        ongoing_reward = 0 
        noise = 0 
        last_reward = 0 
        tracked_noise = []
        while not (terminated or truncated):
            with torch.no_grad():
                observation = observation.reshape((1, 3))
                observation_tensor = torch.from_numpy(observation).to(hyperparameters.device)
                action = policy(observation_tensor)
                action = action.cpu().numpy()
                action += rng.normal(0.0, noise, size=(1,1))
                action = np.clip(action, -2.0, 2.0)
            old_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)
            reward = custom_reward(observation)
            replay_memory.push(old_observation.reshape((1, 3)), action, observation.reshape((1, 3)), reward, terminated)
            for _ in range(5):
                optimize(replay_memory, hyperparameters, policy, policy_target, q_function, q_function_target, policy_optimizer, q_optimizer)
            if (reward < lowest_reward):
                lowest_reward = reward
            noise = get_noise_scaler(reward, last_reward, lowest_reward)
            tracked_noise.append(noise) 
            last_reward = reward 
            soft_weight_update(policy_target, policy, hyperparameters)
            soft_weight_update(q_function_target, q_function, hyperparameters)
            steps_done += 1
            ongoing_reward += reward
        tracked_results.update_decay(steps_done, np.mean(tracked_noise))
        tracked_results.update_rewards(episode, ongoing_reward)

except KeyboardInterrupt:
    pass 

tracked_results.save_model(policy)
tracked_results.save_config()
tracked_results.save_result_graph()
tracked_results.cleanup()


