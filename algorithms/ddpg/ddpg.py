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
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from algorithms.abstract_algorithm import Algorithm
from utils.replay_buffer import *
from utils.ornstein_uhlenbeck_noise import *
from algorithms.ddpg.core import *


class DDPG(Algorithm):
    def __init__(self, runner, num_training_steps, time_per_step, rollout_length=1000, evaluation_length=1000, evaluation_iter=10,
                 num_evaluations=5, random_seed=8, replay_capacity=500, batch_size=100, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3, weight_decay=1e-2, gamma=0.99, tau=0.001, noise_sigma=0.2, noise_theta=0.15,
                 log_path='.', save_path='.', load_path=None):
        """


        The input names are descriptive of variable's meaning, but within the class, the variable names from the paper
        are used.

        :param runner:               (Runner) The interface between the learning agent and the environment.
        :param num_training_steps:   (int)    The total number of steps taken during training. The agent will execute at
                                                least this many steps.
        :param rollout_length:       (int)    The maximum number of steps the agent should take during a rollout.
                                                i.e. how far the agent explores in any given direction.
        :param evaluation_length:    (int)    The maximum number of steps the agent should take when evaluating the
                                                policy.
        :param evaluation_iter:      (int)    The number of exploration iterations completed between policy evaluations.
                                                This is also the point at which the policy is saved.
        :param num_evaluations:      (int)    The number of evaluations that over at an evaluation point. It is best to
                                                complete at lease 3 evaluation traces to account for the variance in
                                                transition probabilities.
        :param random_seed:          (int)    The random seed value to use for the whole training process.
        :param replay_capacity:      (int)
        :param batch_size:           (int)
        TODO: NN architecture info
        :param actor_learning_rate:  (float)
        :param critic_learning_rate: (float)
        :param weight_decay:         (float)
        :param gamma:                (float)
        :param tau:                  (float)
        :param noise_sigma:          (float)
        :param noise_theta:          (float)
        :param log_path:             (str)
        :param save_path:            (str)
        :param load_path:            (str)
        """

        # Save all parameters
        self.runner = runner
        self.is_discrete = runner.is_discrete
        self.H = rollout_length
        self.eval_len = evaluation_length
        self.eval_iter = evaluation_iter
        self.num_evals = num_evaluations
        self.num_training_steps = num_training_steps
        self.save_path = save_path
        self.capacity = replay_capacity
        self.batch_size = batch_size
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.tau = tau

        # Initialize Cuda variables
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Set the random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        # Create the actor and critic neural network
        self.actor = DDPGActor(num_inputs=runner.obs_shape[0], hidden_size1=400, hidden_size2=300,
                               num_actions=runner.action_shape[0], final_bias=3e-3).to(self.device)
        self.critic = DDPGCritic(num_inputs=runner.obs_shape[0], hidden_size1=400, hidden_size2=300,
                                 num_actions=runner.action_shape[0], final_bias=3e-3).to(self.device)

        # Create the target networks
        self.actor_target = DDPGActor(num_inputs=runner.obs_shape[0], hidden_size1=400, hidden_size2=300,
                                      num_actions=runner.action_shape[0], final_bias=3e-3).to(self.device)
        self.critic_target = DDPGCritic(num_inputs=runner.obs_shape[0], hidden_size1=400, hidden_size2=300,
                                        num_actions=runner.action_shape[0], final_bias=3e-3).to(self.device)

        # Create the optimizers for the actor and critic neural networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr,
                                           weight_decay=self.weight_decay)

        # Initialize the target NNs or load models
        if (load_path is None) or (load_path == 'None'):
            # Targets are copied with a hard update
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_target, self.critic)

        else:
            self.load_model(load_path)

        # Create the log files
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        self.log_save_name = log_path + '/episode_performance.csv'
        f = open(self.log_save_name, "w+")
        f.write("training steps, time, steps in evaluation, accumulated reward, done, exit condition \n")
        f.close()

        # Make sure the save path exists
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # Initialize variables used by the learner
        self.replay_buffer = ReplayBuffer(self.capacity)
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(runner.action_shape),
                                                  sigma=(noise_sigma * np.ones(runner.action_shape)),
                                                  theta=noise_theta, dt=time_per_step)

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
        self.runner.reset(evaluate=True)
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
            action = self.get_action(state)  # No noise injected during evaluation

            # Execute determined action
            next_state, reward, done, exit_cond = self.runner.step(action, render=True)

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

        # Get the initial state information and reset the Ornstein-Uhlenbeck noise
        self.noise.reset()
        state = self.runner.get_state()

        # during exploration, update after every step
        while steps_taken < num_steps:

            # Determine the action to take
            action = self.get_action(state, self.noise.noise())

            # Take the action
            next_state, reward, done, exit_cond = self.runner.step(action)

            # Record the information in the replay buffer
            self.replay_buffer.add_memory(state, action, reward, done, exit_cond, next_state)

            # Only update if the replay buffer is full
            if len(self.replay_buffer.rewards) >= self.batch_size:
                # Update the neural networks using a random sampling of the replay buffer
                mb_states, mb_actions, mb_rewards, mb_dones, mb_exits, mb_next_states = self.replay_buffer.sample_batch(
                    self.batch_size)
                self.update_model(mb_states, mb_actions, mb_rewards, mb_dones, mb_exits, mb_next_states)

            # Only reset the environment if a terminal state has been reached
            if done == 1 or exit_cond == 1:
                self.runner.reset()
                self.noise.reset()
                state = self.runner.get_state()
            else:
                state = next_state

            steps_taken += 1

        return

    def get_action(self, state, noise=None):
        """
        This function calculates the desired output using the NN and the exploration noise.

        :input:
            :param state:       (ndarray)   Input state to determine which action to take
            :param noise:       (ndarray)   The exploration noise. Default=[0, 0] (no noise)
        :output:
            :return action:     (ndarray)   The chosen action to take
        """
        # Forward pass the network
        state = torch.FloatTensor(state).to(self.device)
        self.actor.eval()  # Must be in eval mode to execute a forward pass
        action = self.actor.forward(state)
        self.actor.train()  # Must be in train mode to record gradients
        action = action.cpu()

        # Add the process noise to the action
        if noise is not None:
            action = action + torch.FloatTensor(noise).cpu()
        action = action.clamp(-1.0, 1.0)  # Make sure action cannot exceed limits

        # Convert to numpy array
        np_action = action.detach().numpy()

        return np_action

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

        # Evaluate once before training
        t_eval_start = time.time()
        log_time = t_eval_start - t_start - evaluation_time
        avg_steps = 0.0
        avg_reward = 0.0
        for j in range(self.num_evals):
            reward, eval_steps, done, exit_cond = self.evaluate_model(self.eval_len)

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
        self.runner.reset()

        # Iterate through a sufficient number of steps broken into horizons
        while step < self.num_training_steps:

            # Train through exploration
            ep_length = min(self.H, (self.num_training_steps - step))
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
                    reward, eval_steps, done, exit_cond = self.evaluate_model(self.eval_len)

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

    def update_model(self, states, actions, rewards, dones, exits, next_states):
        """
        This function updates neural networks for the actor and critic using back-propogation. More information about
        this process can be found in the DDPG paper.

        :inputs:
            :param states:       (list)  The batch of states from the replay buffer
            :param actions:      (list)  The batch of actions from the replay buffer
            :param rewards:      (list)  The batch of rewards from the replay buffer
            :param dones:        (list)  The batch of done values (1 indicates done, 0 indicates not done) from the
                                           replay buffer
            :param exits:        (list)  The batch of exit conditions (1 indicates exited early, 0 indicates normal)
                                           from the replay buffer
            :param next_states:  (list)  The batch of states reached after executing the actions from the replay buffer
        :outputs:
            :return actor_loss:  (float) The loss value calculated for the actor
            :return critic_loss: (float) The loss value calculated for the critic
        """
        # Convert the inputs into tensors to speed up computations
        batch_states = Variable(torch.FloatTensor(states).to(self.device), requires_grad=True)
        batch_actions = Variable(torch.FloatTensor(actions).to(self.device), requires_grad=True)
        batch_rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        batch_dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        batch_exits = torch.FloatTensor(exits).to(self.device).unsqueeze(1)
        batch_next_states = torch.FloatTensor(next_states).to(self.device)

        # Compute the critics estimated Q values
        batch_qs = self.critic.forward(batch_states, batch_actions)

        # Compute what the actions would have been without noise
        actions_without_noise = self.actor.forward(batch_states)

        # Compute the target's next state and next Q-value estimates used for computing loss
        target_next_action_batch = (self.actor_target.forward(batch_next_states)).detach()
        target_next_q_batch = (self.critic_target.forward(batch_next_states, target_next_action_batch)).detach()

        # Compute y (a metric for computing the critic loss)
        y = (batch_rewards + ((1 - batch_dones) * (1 - batch_exits) * self.gamma * target_next_q_batch)).detach()

        # Compute critic loss and update using the optimizer
        critic_loss = F.mse_loss(y, batch_qs)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss and update using the optimizer
        actor_loss = -self.critic(batch_states, actions_without_noise)
        actor_loss = actor_loss.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, tau=self.tau)
        soft_update(self.critic_target, self.critic, tau=self.tau)

        # new_params = list(self.actor_nn.parameters())[0].clone()
        # print(torch.equal(new_params.data, self.old_params.data))
        #
        # self.old_params = new_params

        # Output the loss values for logging purposes
        return actor_loss.cpu(), critic_loss.cpu()
