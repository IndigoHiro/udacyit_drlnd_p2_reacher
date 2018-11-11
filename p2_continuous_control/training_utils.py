# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:12:10 2018

@author: Hiroyuki.Konno
"""
import random
import numpy as np
import copy
from collections import deque, namedtuple

import torch

class ReplayBuffer:
    """ Fixed size bufffer to store experience tuples"""
    
    def __init__(self, buffer_size, batch_size, seed, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state",
                                                  "action",
                                                  "reward",
                                                  "next_state",
                                                  "done"]
                                     )
        self.seed = random.seed(seed)
        self.device = device
        
    def add(self, states, actions, rewards, next_states, dones):
        assert states.shape[0] == actions.shape[0] and \
            actions.shape[0] == len(rewards) and \
            len(rewards) == next_states.shape[0] and \
            next_states.shape[0] == len(dones)
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        
    def sample_random(self):
        """
        Rondomly sample a batch of experiences from memory and put
        that in device (CPU/GPU)
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.01):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state