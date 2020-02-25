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
    def __init__(self, num_observations, num_actions, runner, step_size, dirs_per_iter, exploration_noise,
                 num_top_performers, rollout_length, evaluation_length, evaluation_iter, num_evaluations,
                 num_training_steps):
        """
        TODO: talk about it
        """

        # Make sure the input parameters are valid
        assert 0 < num_top_performers <= dirs_per_iter, ("num_top_performers: {} is invalid".format(num_top_performers))

        # Save all parameters
        self.runner = runner
        self.alpha = step_size
        self.N = dirs_per_iter
        self.nu = exploration_noise
        self.b = num_top_performers
        self.H = rollout_length
        self.eval_len = evaluation_length
        self.eval_iter = evaluation_iter
        self.num_evals = num_evaluations
        self.num_training_steps = num_training_steps

        # Create the policy
        self.policy = ARSPolicy(num_observations=num_observations, num_actions=num_actions)

    def __do_rollout(self, wieghts):
        """
        TODO
        :param wieghts: (np.ndarray)
        :return:
        """
        # Initialize
        step = 0
        reward_sum = 0

        # Start the evaluation from a safe starting point
        self.runner.reset()
        state = self.runner.get_state()
        done = 0
        exit_cond = 0

        while self.runner.is_available():
            # Stop the controller if there is a collision or time-out
            if done or exit_cond or (step >= self.H != -1):
                # stop
                self.runner.stop()
                break

            # Determine the next action
            action = self.policy.get_action(state, weights=wieghts)

            # Execute determined action
            next_state, reward, done = self.runner.step(action)

            # Update for next step
            reward_sum += reward
            state = next_state
            step += 1

        return reward_sum, step

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
            :return exit_cond:          (int)
        """

        # Initialize
        step = 0
        reward_sum = 0

        # Start the evaluation from a safe starting point
        self.runner.reset()
        state = self.runner.get_state()
        done = 0
        exit_cond = 0

        while self.runner.is_available():
            # Stop the controller if there is a collision or time-out
            if done or exit_cond or (step >= evaluation_length != -1):
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

        return reward_sum, step, done, exit_cond

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
            return final_policy_reward_sum, execution_time, training_time
        """

        # Initialize shit
        t_start = time.time()
        step = 0
        eval_count = 0
        evaluation_time = 0.0

        while step < self.num_training_steps:
            # Sample N noise profiles
            # TODO: figure out how to sample
            noise = np.zeros(self.N)

            # Collect 2N rollouts and their corresponding rewards
            rewards_pos = np.zeros(self.N)
            rewards_neg = np.zeros(self.N)
            for i in range(self.N):
                # Positive rollout
                r_pos, steps_pos = self.__do_rollout(self.policy.theta + (self.nu * noise[i]))

                # Negative rollout
                r_neg, steps_neg = self.__do_rollout(self.policy.theta - (self.nu * noise[i]))

                # Record rewards
                rewards_pos[i] = r_pos
                rewards_neg[i] = r_neg

                # Iterate step count
                step += steps_pos + steps_neg

            # Update the model
            self.update_model(rewards_pos, rewards_neg, noise)

            # Evaluate the model
            if (eval_count + 1) % self.eval_iter == 0:
                t_eval_start = time.time()
                for j in range(self.num_evals):
                    pass  # TODO
                t_eval_end = time.time()
                evaluation_time += t_eval_end - t_eval_start

            # Update normalization parameters in the policy
            self.policy.update_norm()

        t_train = time.time()
        training_time = t_train - t_start - evaluation_time

        final_policy_reward_sum, _, _, _ = self.evaluate_model(self.eval_len)

        t_final = time.time()
        execution_time = t_final - t_start

        return final_policy_reward_sum, execution_time, training_time

    def update_model(self, rewards_pos, rewards_neg, noise):
        """
        This method executes the model update according to the ARS algorithm

        :param rewards_pos: (np.array)
        :param rewards_neg: (np.array)
        :param noise:       (np.array)
        :return:
        """

        # Sort the rewards in descending order according to max{r_pos, r_neg}
        max_r = np.maximum(rewards_pos, rewards_neg)
        indexes = np.argsort(max_r)  # Indexes are arranged for smallest to largest

        sum_augs = 0.0
        l = len(indexes)
        r_2b = []
        for i in range(self.b):
            k = indexes[l - i]
            sum_augs += (rewards_pos[k] - rewards_neg[k]) * noise[k]
            r_2b.append(rewards_pos[k])
            r_2b.append(rewards_neg[k])

        # Calculate the standard deviation of the rewards used for the update. This is used for scaling.
        sigma_r = np.std(np.asarray(r_2b))

        # Compute the new policy weights
        new_policy = self.policy.theta + (self.alpha / (self.b * sigma_r)) * sum_augs

        # Update the policy
        self.policy.theta = new_policy

        return


