#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # self.conv1 = nn.Conv2d(in_channels, 6, 5)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.pool2 = nn.MaxPool2d(4, 4)
        # self.fc1 = nn.Linear(16 * 9 * 9, 256)
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, num_actions)

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4) # dim=(84-(8-1)-1)/4=19        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)          # dim=(19-(4-1)-1)/2=7.5=7
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)          # dim=(7-(2-1)-1)/1=5=5
        self.fc1 = nn.Linear(64 * 7 * 7, 520)
        self.fc2 = nn.Linear(520, num_actions)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # -> n, 4, 84, 84
        # x = self.pool1(F.relu(self.conv1(x)))  # -> n, 6, 40, 40
        # x = self.pool2(F.relu(self.conv2(x)))  # -> n, 16, 9, 9
        # x = x.view(-1, 16 * 9 * 9)            # -> n, 1296
        # x = F.relu(self.fc1(x))               # -> n, 256
        # x = F.relu(self.fc2(x))               # -> n, 64
        # x = self.fc3(x)                       # -> n, 4
        # return x

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*7*7)            # -> n, 1296
        x = self.fc1(x)               # -> n, 256
        x = self.fc2(x)
        return x
        ###########################
        return x
