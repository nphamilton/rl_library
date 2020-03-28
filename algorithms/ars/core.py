"""
File:   core.py
Author: Nathaniel Hamilton

Description: This class implements the associated classes for the Augmented Random Search algorithm written about in
             https://arxiv.org/abs/1803.07055

"""
import numpy as np


class ARSPolicy(object):
    def __init__(self, num_observations, num_actions, discrete):
        """
        This method implements the linear policy described in the original paper and implementation.
        """

        # Initialize the variables accessible to the learning algorithm
        self.theta = np.zeros([num_actions, num_observations], dtype=float)
        self.n = 1.0
        self.mu = np.zeros((num_observations,), dtype=float)
        self.var = np.ones((num_observations,), dtype=float)
        self.std = np.sqrt(self.var)
        self.is_evaluating = True
        self.is_discrete = discrete

        # Initialize the buffer variables for recording and updating the normalizing variables
        self.buffer_n = 0.0
        self.buffer_mu = np.zeros((num_observations,), dtype=float)
        self.buffer_s = np.zeros((num_observations,), dtype=float)

    def get_action(self, state, weights=None):
        """
        This method determines the desired action to take at a given state. Additionally, if the agent is exploring and
        not evaluating, this method records the state information in the form of a running average and standard
        deviation.

        :param state:   (np.array)           The input state
        :param weights: (np.ndarray)         Optional input for following a different set of weights
        :return action: (np.array  or int)   The desired action (will be an index if the action space is discrete
        """

        # Record state info if not evaluating
        if not self.is_evaluating:
            self.__push(state)

        # Normalize the state
        state = self.__normalize(state)

        # Determine the action
        if weights is None:
            action = self.theta.dot(state)
        else:
            assert weights.shape == self.theta.shape, \
                ("weights.shape = {}, theta.shape = {}".format(weights.shape, self.theta.shape))
            action = weights.dot(state)

        # Return the index of the maximum value from the action if the action space is discrete
        if self.is_discrete:
            action = np.argmax(action)

        return action

    def get_noise(self, num_profiles):
        """
        This method returns the specified number of noise profiles. Each the same size as theta.

        :param num_profiles:    (int)        The number of desired noise profiles to be returned
        :return noise_profiles: (np.ndarray) Matrix of random Gaussian distributed values arranged as noise profiles
                                                that are each the same size as the policy weights.
        """

        noise_profiles = np.random.randn(num_profiles, self.theta.shape[0], self.theta.shape[1])

        return noise_profiles

    def __normalize(self, state):
        """
        This method normalizes the input state according to the current mean and standard deviation.

        :param state:   (np.array) The input state
        :return:        (np.array) The normalized state
        """

        return (state - self.mu) / self.std

    def __push(self, state):
        """
        In order to compute a running average and variance during the exploration phase, this method computes and stores
        the values. To incorporate them into the model, update_norm() must be run.

        Method adapted from https://www.johndcook.com/blog/standard_deviation/

        :param state:   (np.array) The input state
        """

        self.buffer_n += 1.0
        if self.buffer_n == 1.0:
            self.buffer_mu = state
            self.buffer_s = np.zeros_like(state)
        else:
            last_mean = self.buffer_mu.copy()
            self.buffer_mu += (state - self.buffer_mu) / self.buffer_n
            self.buffer_s += (state - last_mean) * (state - self.buffer_mu)

        return

    def update_norm(self):
        """
        This method updates the mean and standard deviation for the next round evaluation or exploration
        """

        n1 = self.n
        n2 = self.buffer_n
        n = n1 + n2
        delta = self.mu - self.buffer_mu
        delta2 = delta * delta

        # Compute the new mean
        new_mu = (n1 * self.mu + n2 * self.buffer_mu) / n
        # print('new mu: ' + str(new_mu))

        # Compute the new variance
        old_s = self.var * (self.n - 1)
        old_s[old_s < 0.0] = 0.0
        # print(old_s)
        new_s = old_s + self.buffer_s + delta2 * n1 * n2 / n
        new_var = new_s / (n - 1)
        # print('new var: ' + str(new_var))

        # Update values
        self.n = n
        self.mu = new_mu
        self.var = new_var
        self.std = np.sqrt(new_var)

        # Clear the buffers
        self.buffer_n = 0.0
        self.buffer_mu = np.zeros_like(new_mu)
        self.buffer_s = np.zeros_like(new_s)

        """
        NOTE: This bit is added according to the original implementation found at
        https://github.com/modestyachts/ARS/blob/4c8e24e0a99cf811030e90680fc29eb94fae8cdd/code/filter.py
        """
        # Set values for std less than 1e-7 to +inf to avoid
        # dividing by zero. State elements with zero variance
        # are set to zero as a result.
        # print(self.std)
        self.std[self.std < float(1e-7)] = float("inf")

        return



