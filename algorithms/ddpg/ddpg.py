"""
File:   abstract_algorithm.py
Author: Nathaniel Hamilton

Description:

Usage:          This class should be inherited by each implemented algorithm class in order to enforce consistency.

"""

import time
import gc
import math
import os
import numpy as np
import torch
from algorithms.abstract_algorithm import Algorithm
from utils.replay_buffer import *
# from algorithms.ddpg.core import *


class DDPG(Algorithm):
    def __init__(self):
        i = 1
        i += 1

    def evaluate_model(self, evaluation_length=-1):
        """
        This method performs an evaluation of the model. The evaluation lasts for the specified number of executed
        steps. Multiple evaluations should be used to account for variations in performance.

        :input:
            :param evaluation_length:   (int)   The maximum number of steps to execute during the evaluation run. If -1
                                                    is used, the method will ignore an upper bound for number of steps
                                                    to execute.
        :outputs:
            :return reward_sum:         (float) The cummulative reward collected during the run.
            :return step:               (int)   The number of steps executed during the run.
            :return done:               (int)   A signal indicating if the agent reached a final state.
            :return exit_cond:          (int)   A signal indicating if the agent reached a fatal state.
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
            action = self.policy.get_action(state)  # TODO: correct this

            # Execute determined action
            next_state, reward, done, exit_cond = self.runner.step(action)

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
            load_path
        :output:
            None
        """
        pass

    def save_model(self, save_path):
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
        This method trains the agent to learn an optimal model according to the DDPG algorithm.

        :outputs:
            :return final_policy_reward_sum: (float) The accumulated reward collected by the agent evaluated using the
                                                        latest update of the learned policy.
            :return execution_time:          (float) The amount of time (in seconds) it took to run the full
                                                        train_model() method.
            :return training_time:           (float) The amount of time (in seconds) it took to complete the training
                                                        process.
        """

        # Initialize iterative values
        t_start = time.time()
        step = 0
        eval_count = 0
        evaluation_time = 0.0

        # Iterate through a sufficient number of steps broken into horizons
        while step < self.num_training_steps:

            # Train through exploration
            ep_length = min(self.episode_length, (self.num_training_steps - step))
            self.play_through_training(ep_length)
            step += ep_length

            # Evaluate the model
            eval_count += 1
            if eval_count % self.eval_iter == 0:
                t_eval_start = time.time()
                log_time = t_eval_start - t_start - evaluation_time
                avg_steps = 0.0
                avg_reward = 0.0
                for j in range(self.num_evals):
                    eval_steps, reward, done, exit_cond = self.evaluate_model(self.eval_len)

                    # Log the evaluation run
                    with open(self.log_save_name, "a") as myfile:
                        myfile.write(str(step) + ', ' + str(log_time) + ', ' + str(eval_steps) + ', ' + str(reward) +
                                     ', ' + str(done) + ', ' + str(exit_cond) + '\n')

                    avg_steps += eval_steps
                    avg_reward += reward

                # Print the average results for the user to debugging
                print('Training Steps: ' + str(step) + ', Avg Steps in Episode: ' + str(avg_steps / self.num_evals) +
                      ', Avg Acc Reward: ' + str(avg_reward / self.num_evals))

                # Save the model that achieved this performance
                print('saving...')
                save_path = self.save_path + '/step_' + str(step) + '_model.pth'
                self.save_model(save_path=save_path)

                # Do not count the time taken to evaluate and save the model as training time
                t_eval_end = time.time()
                evaluation_time += t_eval_end - t_eval_start

        t_train = time.time()
        training_time = t_train - t_start - evaluation_time

        # Evaluate and save the final learned model
        final_policy_reward_sum, _, _, _ = self.evaluate_model(self.eval_len)
        print('saving...')
        save_path = self.save_path + '/final_model.pth'
        self.save_model(save_path=save_path)

        t_final = time.time()
        execution_time = t_final - t_start

        return final_policy_reward_sum, execution_time, training_time

    def update_model(self):
        """
        This method executes the model update according to the specific learning algorithm.

        Inputs and outputs for this method depend on the learning algorithm.
        """
        pass
