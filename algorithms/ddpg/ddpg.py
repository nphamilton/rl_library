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
from torch.autograd import Variable
from algorithms.abstract_algorithm import Algorithm
from utils.replay_buffer import *
from algorithms.ddpg.core import *


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

    def __explore(self, num_steps):
        """
        Execute random actions
        :param num_steps:
        :return:
        """

        # Initialize variables
        steps_taken = 0
        done = 0
        exit_cond = 0

        # Get the initial state information and reset the Ornstein-Uhlenbeck noise
        self.noise.reset()
        state, _ = self.runner.get_state()

        # during exploration, update after every step
        while steps_taken < num_steps:

            # Determine the action to take
            action = self.get_action(state, self.noise.noise())

            # Take the action
            next_state, reward, done, exit_cond = self.runner.step(action)

            # Record the information in the replay buffer
            self.replay_buffer.add_memory(state, action, reward, done, next_state)

            # Only update if the replay buffer is full
            if len(self.replay_buffer.rewards) >= self.batch_size:
                # Update the neural networks using a random sampling of the replay buffer
                mb_states, mb_actions, mb_rewards, mb_dones, mb_next_states = self.replay_buffer.sample_batch(
                    self.batch_size)
                actor_loss, critic_loss = self.update_model(mb_states, mb_actions, mb_rewards, mb_dones, mb_next_states)

            # Only reset the environment if a terminal state has been reached
            if done == 1 or exit_cond == 1:
                self.runner.reset()
                self.noise.reset()
                state = self.runner.get_state()
            else:
                state = next_state

            steps_taken += 1

        return

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
            self.__explore(ep_length)
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

    def update_model(self, states, actions, rewards, dones, next_states):
        """
        This function updates neural networks for the actor and critic using back-propogation. More information about
        this process can be found in the DDPG paper.

        :inputs:
            :param states:       (list)  The batch of states from the replay buffer
            :param actions:      (list)  The batch of actions from the replay buffer
            :param rewards:      (list)  The batch of rewards from the replay buffer
            :param dones:        (list)  The batch of done values (1 indicates crash, 0 indicates no crash) from the
                                           replay buffer
            :param next_states:  (list)  The batch of states reached after executing the actions from the replay buffer
        :outputs:
            :return actor_loss:  (float) The loss value calculated for the actor
            :return critic_loss: (float) The loss value calculated for the critic
        """
        # Convert the inputs into tensors to speed up computations
        batch_states = Variable(torch.FloatTensor(states).to(self.device), requires_grad=True)
        batch_actions = Variable(torch.FloatTensor(actions).to(self.device), requires_grad=True)
        batch_rewards = Variable(torch.FloatTensor(rewards).to(self.device), requires_grad=True).unsqueeze(1)
        batch_dones = Variable(torch.FloatTensor(dones).to(self.device), requires_grad=True).unsqueeze(1)
        batch_next_states = Variable(torch.FloatTensor(next_states).to(self.device), requires_grad=True)

        # Compute the critics estimated Q values
        batch_qs = self.critic_nn.forward(batch_states, batch_actions)

        # Compute what the actions would have been without noise
        actions_without_noise = self.actor_nn.forward(batch_states)

        # Compute the target's next state and next Q-value estimates used for computing loss
        target_next_action_batch = (self.actor_target_nn.forward(batch_next_states)).detach()
        target_next_q_batch = (self.critic_target_nn.forward(batch_next_states, target_next_action_batch)).detach()

        # Compute y (a metric for computing the critic loss)
        y = (batch_rewards + ((1 - batch_dones) * self.gamma * target_next_q_batch)).detach()

        # Compute critic loss and update using the optimizer
        critic_loss = F.mse_loss(y, batch_qs)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss and update using the optimizer
        actor_loss = -self.critic_nn(batch_states, actions_without_noise)
        actor_loss = actor_loss.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target_nn, self.actor_nn, tau=self.tau)
        soft_update(self.critic_target_nn, self.critic_nn, tau=self.tau)

        # new_params = list(self.actor_nn.parameters())[0].clone()
        # print(torch.equal(new_params.data, self.old_params.data))
        #
        # self.old_params = new_params

        # Output the loss values for logging purposes
        return actor_loss.cpu(), critic_loss.cpu()
