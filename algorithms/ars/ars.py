"""
File:   ars.py
Author: Nathaniel Hamilton

Description:    This class implements the Augmented Random Search algorithm written about in
                https://arxiv.org/abs/1803.07055

"""
import time
import gc
import os
import numpy as np
import torch
from algorithms.abstract_algorithm import Algorithm
from algorithms.ars.core import *


class ARS(Algorithm):
    def __init__(self, runner, num_training_steps, step_size=0.02, dirs_per_iter=16, num_top_performers=16,
                 exploration_noise=0.03, rollout_length=1000, evaluation_length=1000, evaluation_iter=10,
                 num_evaluations=5, random_seed=8, log_path='.', save_path='.', load_path=None, render_eval=True):
        """
        This class implements the Augmented Random Search algorithm written about in https://arxiv.org/abs/1803.07055
        The default values provided in this class are a blend between the ones used in the authors implementation
        (https://github.com/modestyachts/ARS/tree/4c8e24e0a99cf811030e90680fc29eb94fae8cdd) and another implementation
        like this one that isn't parallelized (https://github.com/colinskow/move37/blob/master/ars/ars.py)

        The input names are descriptive of variable's meaning, but within the class, the variable names from the paper
        are used.

        :param runner:              (Runner) The interface between the learning agent and the environment.
        :param num_training_steps:  (int)    The total number of steps taken during training. The agent will execute at
                                                least this many steps.
        :param step_size:           (float)  A scaling factor for how the policy is adjusted. Value is between 0 and 1.
        :param dirs_per_iter:       (int)    The number of directions to explore at each iteration or exploration.
                                                Value is greater than 1.
        :param num_top_performers:  (int)    The number of top performers to use for computing the update. Value must
                                                be <= dirs_per_iter and > 0.
        :param exploration_noise:   (float)  The standard deviation of the exploration noise applied to each direction.
                                                Value is between 0 and 1.
        :param rollout_length:      (int)    The maximum number of steps the agent should take during a rollout.
                                                i.e. how far the agent explores in any given direction.
        :param evaluation_length:   (int)    The maximum number of steps the agent should take when evaluating the
                                                policy.
        :param evaluation_iter:     (int)    The number of exploration iterations completed between policy evaluations.
                                                This is also the point at which the policy is saved.
        :param num_evaluations:     (int)    The number of evaluations that over at an evaluation point. It is best to
                                                complete at lease 3 evaluation traces to account for the variance in
                                                transition probabilities.
        :param random_seed:         (int)    The random seed value to use for the whole training process.
        :param log_path:            (str)   File path to directory where the 'training_performance.csv' will be saved.
                                                Default='.'
        :param save_path:           (str)   File path to directory where all saved models will be stored. Default='.'
        :param load_path:           (str)   File path to a previously saved model to load from. Default=None indicating
                                                no model to  load from.
        :param render_eval           (bool)   Boolean selection for rendering the environment during evaluation.
        """

        # Make sure the input parameters are valid
        assert 0 < num_top_performers <= dirs_per_iter, ("num_top_performers: {} is invalid".format(num_top_performers))

        # Save all parameters
        self.runner = runner
        self.is_discrete = runner.is_discrete
        self.alpha = step_size
        self.N = dirs_per_iter
        self.nu = exploration_noise
        self.b = num_top_performers
        self.H = rollout_length
        self.eval_len = evaluation_length
        self.eval_iter = evaluation_iter
        self.num_evals = num_evaluations
        self.num_training_steps = num_training_steps
        self.save_path = save_path
        self.render = render_eval

        # Set the random seed
        np.random.seed(random_seed)

        # Create the policy
        self.policy = ARSPolicy(num_observations=runner.obs_shape[0], num_actions=runner.action_shape[0],
                                discrete=self.is_discrete)
        if load_path is not None:
            self.load_model(load_path)

        # Create the log files
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        self.log_save_name = log_path + '/training_performance.csv'
        f = open(self.log_save_name, "w+")
        f.write("training steps, time, steps in evaluation, accumulated reward, done, exit condition \n")
        f.close()

        # Make sure the save path exists
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    def __do_rollout(self, weights):
        """
        This method performs a single rollout using the specified weights.

        :param weights:     (np.ndarray) The policy to follow instead of the trained policy for this rollout
        :return reward_sum: (float)      The total reward accumulated during the rollout
        :return step:       (int)        The number of steps executed during the rollout
        """

        # Initialize
        step = 0
        reward_sum = 0
        self.policy.is_evaluating = False

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
            action = self.policy.get_action(state, weights=weights)

            # Execute determined action
            next_state, reward, done, exit_cond = self.runner.step(action)

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
        self.policy.is_evaluating = True

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
            action = self.policy.get_action(state)

            # Execute determined action
            next_state, reward, done, exit_cond = self.runner.step(action, render=self.render)

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
            :param save_path: (string) The file name the model will be saved to. Default='model.pth'
        """

        # Save everything necessary to start up from this point
        torch.save({
            'theta': self.policy.theta,
            'normalizer_n': self.policy.n,
            'normalizer_means': self.policy.mu,
            'normalizer_vars': self.policy.var
        }, save_path)

        # Clean up any garbage that's accrued
        gc.collect()

        return

    def train_model(self):
        """
        This method trains the agent to learn an optimal model according to the ARS algorithm.

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

        # Evaluate the starting policy
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

        # Train until the maximum number of steps has been reached or passed
        while step < self.num_training_steps:

            eval_count += 1

            # Sample N noise profiles
            noise = self.policy.get_noise(self.N)

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

            # Update normalization parameters in the policy
            self.policy.update_norm()

        t_train = time.time()
        training_time = t_train - t_start - evaluation_time

        # Evaluate and save the final learned model
        final_policy_reward_sum, eval_steps, done, exit_cond = self.evaluate_model(self.eval_len)
        with open(self.log_save_name, "a") as myfile:
            myfile.write(str(step) + ', ' + str(training_time) + ', ' + str(eval_steps) + ', ' +
                         str(final_policy_reward_sum) + ', ' + str(done) + ', ' + str(exit_cond) + '\n')
        print('saving...')
        save_path = self.save_path + '/final_model.pth'
        self.save_model(save_path=save_path)

        t_final = time.time()
        execution_time = t_final - t_start

        return final_policy_reward_sum, execution_time, training_time

    def update_model(self, rewards_pos, rewards_neg, noise):
        """
        This method executes the model update according to the ARS algorithm

        :param rewards_pos: (np.array) An array of the rewards collected after adding the noise to the policy weights
        :param rewards_neg: (np.array) An array of the rewards collected after subtracting the noise from the policy
                                        weights
        :param noise:       (np.array) An array of the noise
        :return:
        """

        # Sort the rewards in descending order according to max{r_pos, r_neg}
        max_r = np.maximum(rewards_pos, rewards_neg)
        # print(max_r)
        indexes = np.argsort(max_r)  # Indexes are arranged from smallest to largest

        sum_augs = np.zeros_like(self.policy.theta)
        l = len(indexes) - 1
        r_2b = []
        for i in range(self.b):
            k = indexes[l - i]
            sum_augs += (rewards_pos[k] - rewards_neg[k]) * noise[k]
            r_2b.append(rewards_pos[k])  # - rewards_neg[k])
            r_2b.append(rewards_neg[k])

        # Calculate the standard deviation of the rewards used for the update. This is used for scaling.
        sigma_r = np.std(np.asarray(r_2b))
        # As this value gets smaller when approaching convergence, make sure there is no divide by 0
        if sigma_r > float(1e-7):
            sigma_r = sigma_r
        else:
            sigma_r = float("inf")
        # print((self.alpha / (self.b * sigma_r)) * sum_augs)

        # Compute the new policy weights
        new_policy = self.policy.theta + ((self.alpha / (self.b * sigma_r)) * sum_augs)
        # print(new_policy)

        # Update the policy
        self.policy.theta = new_policy

        return
