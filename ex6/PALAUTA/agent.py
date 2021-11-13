import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 16
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        # TODO: Add another linear layer for the critic
        self.fc2_critic = torch.nn.Linear(self.hidden, 1)
        #self.sigma = torch.zeros(1)  # TODO: Implement learned variance (or copy from Ex5)
        self.sigma = torch.nn.Parameter(torch.tensor([10.0]))  #initial variance = 100
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Common part
        x = self.fc1(x)
        x = F.relu(x)

        # Actor part
        action_mean = self.fc2_mean(x)
        sigma = self.sigma  # TODO: Implement (or copy from Ex5)

        # Critic part
        # TODO: Implement
        critic_value = self.fc2_critic(x)

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma
        # Implement or copy from Ex5
        action_dist = Normal(action_mean, torch.abs(sigma)+0.1)

        # TODO: Return state value in addition to the distribution

        return critic_value, action_dist


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []

    def update_policy(self, episode_number):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []

        # TODO: Compute state values
        num_rewards = rewards.size(dim=0)

        state_values = torch.zeros(num_rewards)
        for sidx in torch.arange(num_rewards):
            critic_value, _ = self.policy(states[sidx])
            state_values[sidx] = critic_value
        
        next_state_values = torch.zeros(num_rewards)
        for sidx in torch.arange(num_rewards):
            critic_value, _ = self.policy(next_states[sidx])   
            if done[sidx]:
                next_state_values[sidx] = torch.tensor([0]).float().to(self.train_device)
            else:
                next_state_values[sidx] = critic_value
                
        # TODO: Compute critic loss (MSE)
        TD_target = rewards + self.gamma * next_state_values
        TD_target = TD_target.detach()
        critic_loss = F.mse_loss(state_values, TD_target)

        # Advantage estimates
        # TODO: Compute advantage estimates        
        advantage = rewards + self.gamma * next_state_values - state_values
        advantage = advantage.detach()
        
        # TODO: Calculate actor loss (very similar to PG)
        actor_loss = (-action_probs * advantage).mean()
        
        # TODO: Compute the gradients of loss w.r.t. network parameters
        # Or copy from Ex5
        loss =  actor_loss + critic_loss
        
        # TODO: Update network parameters using self.optimizer and zero gradients
        # Or copy from Ex5
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network
        # Or copy from Ex5
        _, action_dist = self.policy(x)

        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy
        # Or copy from Ex5
        if evaluation:
            action = action_dist.mean()
        else:
            action = action_dist.sample()            

        # TODO: Calculate the log probability of the action
        # Or copy from Ex5
        act_log_prob = action_dist.log_prob(action)

        return action, act_log_prob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        # Now we need to store some more information than with PG
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
