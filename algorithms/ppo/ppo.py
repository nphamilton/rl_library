"""
File:   ppo.py
Author: Nathaniel Hamilton

Description:    This class implements the Proximal Policy Optimization algorithm written about in
                https://arxiv.org/abs/1707.06347

"""

import time
import gc
import math
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from algorithms.abstract_algorithm import Algorithm
from algorithms.ppo.core import *


class PPO(Algorithm):
    def __init__(self):
        """

        """

    def evaluate_model(self, evaluation_length=-1):
        """
        This method performs an evaluation of the model. The evaluation lasts for the specified number of executed
        steps. Multiple evaluations should be used to account for variations in performance.

        :input:
            evaluation_length
        :output:
            return reward_sum, steps, done, exit_cond
        """
        pass

    def load_model(self, load_path=None):
        """
        This method loads a model. The loaded model can be a previously learned policy or an initializing policy used
        for consistency.

        :input:
            load_path
        :output:
            None
        """
        pass

    def save_model(self, save_path='.'):
        """
        This method saves the current model learned by the agent.

        :input:
            save_path
        :output:
            None
        """
        pass

    def train_model(self):
        """
        This method trains the agent to learn an optimal model in the manner specified by the learning algorithm.

        :input:
            None (all parameters and hyperparameters should be specified upon initialization
        :output:
            return final_policy_reward_sum, execution_time
        """
        pass

    def update_model(self):
        """
        This method executes the model update according to the specific learning algorithm.

        Inputs and outputs for this method depend on the learning algorithm.
        """
        pass
