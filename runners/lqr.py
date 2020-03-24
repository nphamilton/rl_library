"""
File:   lqr.py
Author: Nathaniel Hamilton

Description:    This runner...TODO


"""
import numpy as np

from runners.abstract_runner import Runner


class LQRRunner(Runner):
    def __init__(self, state_matrix=np.array([[1.01, 0.01, 0.0], [0.01, 1.01, 0.01], [0., 0.01, 1.01]]),
                 input_matrix=np.eye(3), state_cost=(np.ones(3) / 1000), input_cost=np.ones(3), cross_term=None,
                 horizon_length=300, evaluation_init=np.array([1.33828699, -2.61368526, -1.85276285])):
        """

        :param state_matrix:        (np.ndarray) nxn matrix
        :param input_matrix:        (np.ndarray) nxm matrix
        :param state_cost:          (np.array)   n length array
        :param input_cost:          (np.array)   m length array
        :param cross_term:
        :param horizon_length:
        :param evaluation_init:
        """

        # Save the parameters
        self.A = state_matrix
        self.B = input_matrix
        self.Q = np.diag(state_cost)
        self.R = np.diag(input_cost)
        self.horizon_length = horizon_length
        self.eval_init = evaluation_init

        if cross_term is None:
            self.N = np.zeros_like(input_matrix)

        # Establish the required variables for learning algorithms to access
        self.obs_shape = np.array([len(state_cost)])
        self.action_shape = np.array([len(input_cost)])

        # Initialize variables
        self.state = None
        self.time = 0

    def get_state(self):
        """
        This function should return the current state of the agent in the environment. No reward or status of done or
        exit will be provided, just the state.

        """
        return self.state

    def is_available(self):
        """
        This environment is always available, so returns True.
        """
        return True

    def step(self, action):
        """
        This function executes as single step of the LQR system. For more information about how a LQR step works, look
        at the Wikipedia article https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator

        :input:
            :param action:  (np.array)
        :outputs:
            :return next_state:
            :return reward:
            :return done:
            :return exit_cond:
        """

        # Record the state
        x = self.state

        # Compute the next state and the accumulated cost
        x_next = np.dot(self.A, x) + np.dot(self.B, action)
        cost = np.dot(x.T, np.dot(self.Q, x)) + np.dot(action.T, np.dot(self.R, action)) + 2 * \
               np.dot(x.T, np.dot(self.N, action))
        self.time += 1

        # Determine if the state is terminal
        done = 0
        if self.time >= self.horizon_length:
            done = 1

        # Convert values to outputs
        next_state = x_next
        reward = 1 * cost
        exit_cond = 0

        return next_state, reward, done, exit_cond

    def stop(self):
        """
        This method is meant to stop a simulation from continuing. Since this simulation only continues through a step,
        the stop method does not do anything.
        """
        return

    def reset(self, evaluate=False):
        """
        This function resets the environment to a random starting point.

        :param evaluate:    (bool) When this is true, the environment should reset to set starting point for consistent
                                    evaluation
        """

        if eval:
            self.state = self.eval_init
            self.time = 0
        else:
            self.state = np.random.normal(0, 1, size=self.obs_shape)
            self.time = 0

        return
