import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        #self.sigma = torch.zeros(1)  # TODO: Implement accordingly (T1, T2)
        self.sigma = torch.tensor([5.0])  #T1: variance=25
        #self.sigma = torch.tensor([10.0])  #T2: variance=100
        #self.sigma = torch.nn.Parameter(torch.tensor([10.0]))  #T2: variance learned as parameter, initial variance=100
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        sigma = self.sigma  # TODO: Is it a good idea to leave it like this?

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        action_dist = Normal(action_mean, sigma)
        
        return action_dist

    def set_sigma(self, value):  #added for T2
        self.sigma = torch.tensor([value])


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # TODO: Compute discounted rewards
        num_rewards = rewards.size(dim=0)

        discounted_rewards = torch.zeros(num_rewards)
        for ridx in torch.arange(num_rewards):
            discounted_rewards[ridx] = self.gamma**ridx * rewards[ridx]
        
        discounted_rewards = torch.flip(discounted_rewards, [0])
        discounted_rewards = torch.cumsum(discounted_rewards, dim=0)
        discounted_rewards = torch.flip(discounted_rewards, [0])

        # subtract baseline from the discounted rewards
        std = 1
        #b = 0   #T1-(a) b = 0
        b = 20  #T1-(b) b = 20
        #b = torch.mean(discounted_rewards)  #T1-(c) b = mean(*)        
        #std = torch.std(discounted_rewards)

        discounted_rewards = (discounted_rewards - b) / std
        
        # T2-(a): set variance for the next episode
        #print("episode: {} -> sigma: {}".format(episode_number, self.policy.sigma))
        #c = 5e-4
        #variance = 100 * np.exp(-c * episode_number)
        #self.policy.set_sigma(np.sqrt(variance))

        # TODO: Compute the optimization term (T1)
        self.optimizer.zero_grad()
        opt_term = discounted_rewards * action_probs

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss = -opt_term.mean()
        loss.backward()
        
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()


    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        action_dist = self.policy(x)
        
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = action_dist.mean()
        else:
            action = action_dist.sample()            

        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = action_dist.log_prob(action)

        return action, act_log_prob


    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

