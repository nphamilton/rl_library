"""
File:   ppo.py
Author: Nathaniel Hamilton

Description:    This class implements the Proximal Policy Optimization algorithm written about in
                https://arxiv.org/abs/1707.06347

"""

import time
import gc
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from algorithms.abstract_algorithm import Algorithm
from algorithms.ppo.core import *


class PPO(Algorithm):
    def __init__(self, runner, num_training_steps, rollout_length=2048, evaluation_length=2048,
                 evaluation_iter=10, num_evaluations=5, random_seed=8, minibatch_size=64, num_epochs=10,
                 learning_rate=3e-4, discount_gamma=0.995, gae_lambda=0.97, clip_param=0.2,
                 log_path='.', save_path='.', load_path=None, render_eval=True):
        """
        This class implements the Proximal Policy Optimization algorithm written about in
        https://arxiv.org/abs/1707.06347

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
        TODO NN architecture
        :param log_path:             (str)    File path to directory where episode_performance.csv will be saved.
        :param save_path:            (str)    File path to directory where all models will be saved.
        :param load_path:            (str)    File path to initial model to be loaded.
        :param render_eval           (bool)   Boolean selection for rendering the environment during evaluation.
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
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.render = render_eval

        # Initialize Cuda variables
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Set the random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        # Create the actor and critic neural network
        if self.is_discrete:
            self.actor = DiscreteActor(num_inputs=runner.obs_shape[0], hidden_size1=64, hidden_size2=64,
                                       num_actions=runner.action_shape[0]).to(self.device)
            self.num_actions = runner.action_shape[0]
        else:
            self.actor = ContinuousActor(num_inputs=runner.obs_shape[0], hidden_size1=64, hidden_size2=64,
                                         num_actions=runner.action_shape[0]).to(self.device)
            self.num_actions = None
        self.critic = PPOCritic(num_inputs=runner.obs_shape[0], hidden_size1=64, hidden_size2=64).to(self.device)

        # Create the optimizers for the actor and critic neural networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the target NNs or load models
        if load_path is not None:
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
        self.old_params = list(self.actor.parameters())[0].clone()

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
            action, _, _ = self.get_action_and_value(state, evaluate=True)

            # Execute determined action
            next_state, reward, done, exit_cond = self.runner.step(action, render=self.render)

            # Update for next step
            reward_sum += reward
            state = next_state
            step += 1

        return reward_sum, step, done, exit_cond

    def __explore(self, num_steps):
        """

        :param num_steps:
        :return:
        """

        # Initialize
        step = 0
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        # Reset the environment for a new round of exploration
        self.runner.reset()
        state = self.runner.get_state()
        done = 0
        exit_cond = 0

        while self.runner.is_available() and step < num_steps:
            # Stop and reset if a terminal condition is reached
            if done or exit_cond:
                # Stop the runner
                self.runner.stop()

                # Reset the environment for a new round of exploration
                self.runner.reset()
                state = self.runner.get_state()

            # Determine the next action
            action, log_prob, val = self.get_action_and_value(state)

            # Execute determined action
            next_state, reward, done, exit_cond = self.runner.step(action)

            # Record info
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(max((done + exit_cond), 1))
            values.append(val)

            # Update for next step
            state = next_state
            step += 1

        # Backwards compute the returns using the Generalized Advantage Estimation method
        _, _, next_value = self.get_action_and_value(state)
        returns, advantages = compute_gae(next_value=next_value, rewards=rewards, values=values, dones=dones,
                                          gamma=self.discount_gamma, lam=self.gae_lambda)

        return states, actions, log_probs, returns, advantages, values

    def get_action_and_value(self, state, evaluate=False):
        """

        :param state:
        :param evaluate:
        :return np_action:
        :return log_prob:
        :return value:
        """

        # Initialize the log probability in case it isn't computed
        log_prob = 0.0

        # Selecting an action is different depending on the action space
        if self.is_discrete:
            # Forward pass the network
            state = torch.FloatTensor(state).to(self.device)
            policy_dist, _ = self.actor.forward(state)
            policy_dist = policy_dist.cpu()
            distribution = policy_dist.detach().numpy()

            if evaluate:
                np_action = np.argmax(distribution)
            else:
                # Choose a random action according to the distribution
                np_action = np.random.choice(self.num_actions, p=np.squeeze(distribution))

                # Calculate the log probability of choosing that action
                log_prob = torch.log(policy_dist.squeeze(0)[np_action])

        else:
            # Forward pass the network
            state = torch.FloatTensor(state).to(self.device)
            mu, std, _ = self.actor.forward(state)
            means = mu.cpu()
            stds = std.cpu()

            if evaluate:
                np_action = means.detach().numpy()
            else:
                # Choose a random action according to the distribution
                random_val = torch.FloatTensor(np.random.rand(2)).cpu()
                action = (means + stds * random_val)
                np_action = action.detach().numpy()

                # Calculate the log probability of choosing that action
                log_prob = calculate_log_probability(x=action, mu=means, std_devs=stds)

        # Compute the estimated value of the current state/observation
        val = self.critic.forward(state)
        value = val.detach().numpy()[0]

        return np_action, log_prob, value

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
        This method trains the agent to learn an optimal model according to the PPO algorithm.

        :outputs:
            :return final_policy_reward_sum: (float) The accumulated reward collected by the agent evaluated using the
                                                        latest update of the learned policy.
            :return execution_time:          (float) The amount of time (in seconds) it took to run the full
                                                        train_model() method.
            :return training_time:           (float) The amount of time (in seconds) it took to complete the training
                                                        process.
        """

        # Initialize iterative values
        self.actor.train()
        self.critic.train()
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

            # Explore through a horizon
            ep_length = min(self.H, (self.num_training_steps - step))
            h_states, h_actions, h_log_probs, h_returns, h_advs, h_values = self.__explore(ep_length)
            step += ep_length

            # Iterate through the number of epochs, running updates on shuffled minibatches
            indices = np.arange(ep_length)
            for e in range(self.num_epochs):
                # Shuffle the frames in the horizon
                np.random.shuffle(indices)

                # Update each shuffled minibatch (mb)
                for mb_start in range(0, ep_length, self.minibatch_size):
                    # Single out each minibatch from the recorded horizon
                    mb_end = mb_start + self.minibatch_size
                    mb_indices = indices[mb_start:mb_end]
                    mb_states = []
                    mb_actions = []
                    mb_log_probs = []
                    mb_returns = []
                    mb_advs = []
                    mb_values = []
                    for i in mb_indices:
                        mb_states.append(h_states[i])
                        mb_actions.append(h_actions[i])
                        mb_log_probs.append(h_log_probs[i])
                        mb_returns.append(h_returns[i])
                        mb_advs.append(h_advs[i])
                        mb_values.append(h_values[i])

                    # print('updating...')
                    # Update the shuffled minibatch
                    actor_loss, critic_loss = self.update_model(mb_states, mb_actions, mb_log_probs, mb_returns,
                                                                mb_advs, mb_values)
                    # print(f'Actor loss: {actor_loss}, Critic Loss: {critic_loss}')

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

    def update_model(self, states, actions, old_log_probs, returns, advantages, old_values):
        """
        This function updates neural networks for the actor and critic using back-propogation. More information about
        this process can be found in the PPO paper (https://arxiv.org/abs/1707.06347) and the pytorch implementation
        found at https://github.com/higgsfield/RL-Adventure-2

        :inputs:
            :param states:          (list)  The observations recorded during the horizon.
            :param actions:         (list)  The actions taken during the horizon.
            :param old_log_probs:   (list)  The log probability of each action calculated during the horizon.
            :param returns:         (list)  The returns calculated during the horizon.
            :param old_values:      (list)  The estimate state values acquired during the horizon.
        :outputs:
            :return actor_loss:     (float) The loss value calculated for the actor.
            :return critic_loss:    (float) The loss value calculated for the critic.
        """

        # Calculate new values and log probabilities
        states = Variable(torch.FloatTensor(states).to(self.device), requires_grad=True)
        new_log_probs = []
        if self.is_discrete:
            # actions = Variable(torch.LongTensor(actions).to(self.device), requires_grad=True)
            for i in range(len(actions)):
                _, log_prob = self.actor.forward(states[i], actions[i])
                new_log_probs.append(log_prob)
        else:
            actions = Variable(torch.FloatTensor(actions).to(self.device), requires_grad=True)
            for i in range(len(actions)):
                _, _, log_prob = self.actor.forward(states[i], actions[i])
                new_log_probs.append(log_prob)
        new_values = self.critic.forward(states)

        # print(new_log_probs)

        # Convert arrays into tensors and send them to the GPU to speed up calculations
        old_values = torch.FloatTensor(old_values).detach().to(self.device)
        new_values = new_values.to(self.device)
        returns = torch.FloatTensor(returns).detach().to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).detach().to(self.device)
        new_log_probs = torch.stack(new_log_probs).to(self.device)

        # print(new_log_probs)

        # Calculate the advantage and normalize it
        # advantages = returns - old_values
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Compute the actor loss function with clipping
        ratio = (new_log_probs - old_log_probs).exp()
        loss_not_clipped = ratio * advantages
        loss_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(loss_not_clipped, loss_clipped).mean()

        # Update actor using actor optimizer
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute the critic loss function with clipping
        # clipped_values = old_values + torch.clamp((new_values - old_values), -self.clip_param, self.clip_param)
        # closs_clipped = (returns - clipped_values).pow(2)
        # closs_not_clipped = (returns - new_values).pow(2)
        # critic_loss = 0.5 * torch.max(closs_clipped, closs_not_clipped).mean()
        critic_loss = (returns - new_values).pow(2).mean()

        # Update critic using critic optimizer
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # new_params = list(self.actor.parameters())[0].clone()
        # print(torch.equal(new_params.data, self.old_params.data))
        #
        # self.old_params = new_params

        # Output the loss values for logging purposes
        return actor_loss.cpu(), critic_loss.cpu()
