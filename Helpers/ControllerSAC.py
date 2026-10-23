
import torch
import numpy as np 

from Helpers.Memory import Memory
from Helpers.Hyperparameters import HyperparameterData

from Helpers.ControllerDDPG import get_eps
from Helpers.ControllerDDPG import hard_weight_update, soft_weight_update

# Assume 4 layers
class PolicySAC(torch.nn.Module):
    def __init__(self, hyperparameters : HyperparameterData):
        super().__init__()
        self.device = hyperparameters.device
        sizes = hyperparameters.neurons.copy()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(sizes[0], sizes[1]),
            torch.nn.LayerNorm(sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(sizes[1], sizes[2]),
            torch.nn.ReLU()

        ).to(self.device)
        self.mean_layer = torch.nn.Linear(sizes[2], sizes[3]).to(self.device)
        torch.nn.init.uniform_(self.mean_layer.weight, -0.0001, 0.0001)
        torch.nn.init.uniform_(self.mean_layer.bias, -0.0001, 0.0001)
        self.std_layer = torch.nn.Linear(sizes[2], sizes[3]).to(self.device)
        torch.nn.init.uniform_(self.std_layer.weight, -0.0001, 0.0001)
        torch.nn.init.uniform_(self.std_layer.bias, -0.0001, 0.0001)

    def forward(self, x):
        x = x.to(self.device)
        logits = self.linear_relu_stack(x)
        mean = self.mean_layer(logits)
        std = self.std_layer(logits)
        std = torch.clamp(std, -20, 2)
        return mean, torch.exp(std)

class SoftQ_SAC(torch.nn.Module):
    def __init__(self, hyperparameters : HyperparameterData):
        super().__init__()
        self.device = hyperparameters.device
        sizes = hyperparameters.neurons.copy()
        sizes[0] += sizes[-1]
        sizes[-1] = 1
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(sizes[0], sizes[1]),
            torch.nn.LayerNorm(sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(sizes[1], sizes[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(sizes[2], sizes[3])

        ).to(self.device)

    def forward(self, x1, x2):
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        x = torch.cat((x1, x2), dim=1)
        logits = self.linear_relu_stack(x)
        return logits


def get_action_array(policy : PolicySAC, hyperparameters : HyperparameterData, observation):
    with torch.no_grad():
        observation_tensor = torch.tensor(observation)
        means, std = policy(observation_tensor)
        normal = torch.distributions.Normal(means, std)
        action = normal.sample()
        output = torch.tanh(action)
        return output.cpu().detach().numpy(), std.cpu().detach().numpy()

def optimizeSAC(sample_memory : Memory, 
                hyperparameters : HyperparameterData, 
                soft_q_functions : list,
                soft_q_target_functions : list,
                policy_function, 
                policy_optimizer, 
                softQ_optimizers : list,
                alpha_log,
                alpha_optimizer):  

    # Set up hyperparameters
    batch_size = hyperparameters.batch_size
    if len(sample_memory.states) < batch_size:
        return 
    device = hyperparameters.device
    gamma = hyperparameters.gamma

    # Extract and Convert Sample
    sample = sample_memory.sample(batch_size)
    old_states = torch.tensor(np.array(sample[0])).to(device).float()
    action = torch.tensor(np.array(sample[1])).to(device).float()
    states = torch.tensor(np.array(sample[2])).to(device).float()
    rewards = torch.tensor(sample[3]).to(device).float().unsqueeze(1)
    dones = torch.tensor(sample[4], dtype=torch.bool).to(device).unsqueeze(1)

    # Calculate Alpha 
    # Calculate pi(a_t|s_t)
    mean, std = policy_function(old_states)
    normal = torch.distributions.Normal(mean, std)
    normal_samples = normal.rsample()
    action_samples = torch.tanh(normal_samples)

    # Compute log of policy output
    log_normal_samples = normal.log_prob(normal_samples)
    log_normal_samples -= torch.log(1 - action_samples.pow(2) + 1e-6)
    log_normal_samples = log_normal_samples.sum(dim=1, keepdim=True)

    # Setup for alpha usage
    alpha = alpha_log.exp()
    alpha_detached = alpha.detach()

    # Calculate loss using -alpha(log(pi(a_t|s_t)) + H)
    alpha_loss = (-alpha*(log_normal_samples.detach() - 1.0)).mean()

    # Optimize alpha
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    # Calculate soft Q Loss
    with torch.no_grad():
        # First, find V(s) using formula
        # Calculate log(pi(s_{t+1}))
        next_mean, next_std = policy_function(states)
        next_normal = torch.distributions.Normal(next_mean, next_std)
        next_normal_samples = next_normal.rsample()
        next_action_samples = torch.nn.Tanh()(next_normal_samples)
        next_log_normal_samples = next_normal.log_prob(next_normal_samples)
        next_log_normal_samples -= torch.log(1 - next_action_samples.pow(2) + 1e-6)
        next_log_normal_samples = next_log_normal_samples.sum(dim=1, keepdim=True)

        # Calculate Q(s_{t+1}, a_{t+1})
        qs = []
        for qModel in soft_q_target_functions:
            # Used s_{t+1} and a_{t+1}
            # a_{t+1} uses the sample not the mean
            # to add entropy 
            qs.append(qModel(states, next_action_samples))
        stacked_qs = torch.stack(qs)
        q_minimum, _ = torch.min(stacked_qs, dim=0)
        q_minimum = q_minimum.view(-1, 1)

        # Find V(s) using the minimum
        v = q_minimum - alpha_detached*log_normal_samples
        v[dones] = 0.0

        # Use Bellman for Q^ 
        q_hat = rewards + gamma*v

    # Calculate Q(s, a)
    qs = []
    for qModel in soft_q_functions:
        qs.append(qModel(old_states, action))

    # Calculate loss for Q and Q^
    losses_counter = 0
    for q in qs:
        losses_counter += torch.nn.MSELoss()(q, q_hat)

    # Optimize Q Functions
    for optim in softQ_optimizers:
        optim.zero_grad()
    losses_counter.backward()
    for optim in softQ_optimizers:
        optim.step()

    # Calculate Q(s_t, f(epsilon;s_t))
    qs = []
    for qModel in soft_q_functions:
        qs.append(qModel(old_states, action_samples))
    qs_stacked = torch.stack(qs)
    q_minimum, _ = torch.min(qs_stacked, dim=0)

    # Calculate loss of policy 
    policy_loss = (alpha_detached*log_normal_samples - q_minimum).mean()

    # Optimize Policy
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
