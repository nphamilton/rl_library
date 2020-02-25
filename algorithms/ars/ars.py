"""
File:   ars.py
Author: Nathaniel Hamilton

Description:

Usage:       This class implements the Augmented Random Search algorithm written about in
             https://arxiv.org/abs/1803.07055

"""
import time
import gc
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from algorithms.abstract_algorithm import Algorithm
from algorithms.ars.core import *


class ARS(Algorithm):
    def __init__(self):
        """
        TODO: talk about it
        """

        # Save all parameters

        # Create the policy
        self.policy = ARSPolicy(num_observations=num_observations, num_actions=num_actions)

    def evaluate_model(self, evaluation_length=-1):
        """
        This method performs an evaluation of the model. The evaluation lasts for the specified number of executed
        steps. Multiple evaluations should be used to account for variations in performance.

        :input:
            :param evaluation_length:   (int)
        :outputs:
            :return reward_sum:         (float)
            :return step:               (int)
            :return done:               (int)
            :return exit:               (int)
        """

        # Initialize
        step = 0
        reward_sum = 0

        # Start the evaluation from a safe starting point
        self.runner.reset()
        state = self.runner.get_state()
        done = 0
        exit = 0

        while self.runner.is_available():
            # Stop the controller if there is a collision or time-out
            if done or exit or (step >= evaluation_length != -1):
                # stop
                self.runner.stop()
                break

            # Determine the next action
            action = self.policy.get_action(state)

            # Execute determined action
            next_state, reward, done = self.runner.step(action)

            # Update for next step
            reward_sum += reward
            state = next_state
            step += 1

        return reward_sum, step, done, exit

    def load_model(self, load_path):
        """
        This method loads a model. The loaded model can be a previously learned policy or an initializing policy used
        for consistency.

        :input:
            :param load_path: (string) The file name where the models will be loaded from. default=None
        """

        # Load the saved file as a dictionary
        if load_path is not None:
            checkpoint = torch.load(load_path)

            self.policy.theta = checkpoint['theta']
            self.policy.n = checkpoint['normalizer_n']
            self.policy.mu = checkpoint['normalizer_means']
            self.policy.var = checkpoint['normalizer_vars']

        # Clean up any garbage that's accrued
        gc.collect()

        return

    def save_model(self, save_path='model.pth'):
        """
        This method saves the current model learned by the agent.

        :input:
            save_path
        :output:
            None
        """

        # Save everything necessary to start up from this point
        torch.save({
            'theta': self.policy.theta,
            'normalizer_n': self.policy.n,
            'normalizer_means': self.policy.mu,
            'normalizer_vars': self.policy.var
        })

        # Clean up any garbage that's accrued
        gc.collect()

        return

    def train_model(self):
        """
        This method trains the agent to learn an optimal model according to the ARS algorithm.

        :output:
            return final_policy_reward_sum, execution_time
        """
        pass

    def update_model(self):
        """
        This method executes the model update according to the ARS algorithm

        :input:
            evaluation_length
        :output:
            return reward_sum, steps, done, exit
        """
        pass
