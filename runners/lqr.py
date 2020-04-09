"""
File:   lqr.py
Author: Nathaniel Hamilton

Description:    This runner simulates a Linear Quadratic Regression (LQR) system.


"""
import numpy as np

from runners.abstract_runner import Runner


class LQRRunner(Runner):
    def __init__(self, state_matrix=np.matrix([[1.01, 0.01, 0.0], [0.01, 1.01, 0.01], [0., 0.01, 1.01]]),
                 input_matrix=np.matrix(np.eye(3)), state_cost=(1e-3 * np.ones(3)), input_cost=np.ones(3),
                 cross_term=None, max_action=np.array([1, 1, 1]), min_action=None,
                 max_state=np.array([1000., 1000., 1000.]), min_state=None, scale=1, horizon_length=300,
                 max_init_state=np.array([3., 3., 3.]), min_init_state=None,
                 evaluation_init=np.array([1.33828699, -2.61368526, -1.85276285])):
        """

        :param state_matrix:        (np.matrix) nxn matrix
        :param input_matrix:        (np.matrix) nxm matrix
        :param state_cost:          (np.array)   n length array
        :param input_cost:          (np.array)   m length array
        :param cross_term:
        :param horizon_length:
        :param evaluation_init:
        """

        # Save the parameters
        self.A = state_matrix
        self.B = input_matrix
        self.Q = np.matrix(np.diag(state_cost))
        self.R = np.matrix(np.diag(input_cost))
        self.horizon_length = horizon_length
        self.eval_init = evaluation_init
        self.scale = scale

        self.max_action = max_action
        if min_action is None:
            self.min_action = -max_action
        else:
            self.min_action = min_action

        self.max_state = max_state
        if min_state is None:
            self.min_state = -max_state
        else:
            self.min_state = min_state

        self.max_init = max_init_state
        if min_init_state is None:
            self.min_init = -1 * max_init_state
        else:
            self.min_init = min_init_state

        if cross_term is None:
            self.N = np.matrix(np.zeros_like(input_matrix))

        # Establish the required variables for learning algorithms to access
        self.obs_shape = np.array([len(state_cost)])
        # print(f'lqr.obs_shape: {state_cost}') #{self.obs_shape}')
        self.action_shape = np.array([len(input_cost)])
        # print(f'lqr.action_shape: {self.action_shape}')
        self.is_discrete = False

        # Initialize variables
        self.state = np.matrix(np.zeros_like(self.max_state)).T
        self.scale_mult = (self.max_action - self.min_action) / 2.0
        self.scale_add = (self.max_action - self.min_action) / 2.0 + self.min_action
        self.time = 0

    def get_state(self):
        """
        This function should return the current state of the agent in the environment. No reward or status of done or
        exit will be provided, just the state.

        """
        return np.asarray(self.state.T)

    def is_available(self):
        """
        This environment is always available, so returns True.
        """
        return True

    def step(self, action, render=False):
        """
        This function executes as single step of the LQR system. For more information about how a LQR step works, look
        at the Wikipedia article https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator

        :input:
            :param action:  (np.array)
            :param render:  (bool)
        :outputs:
            :return next_state:
            :return reward:
            :return done:
            :return exit_cond:
        """

        # Record the state
        x = self.state

        # Adjust the action
        if self.scale == 1:
            # Scale the action
            action = np.matrix(np.multiply(action, self.scale_mult) + self.scale_add)
        elif self.scale == 0:
            action = np.matrix(np.minimum(np.maximum(action, self.min_action), self.max_action))
        else:
            raise NotImplementedError
        action = action.T

        # Compute the next state and the accumulated cost
        # print(f'A: {self.A}')
        # print(f'x: {x}')
        x_next = np.matmul(self.A, x) + np.matmul(self.B, action)  # + np.random.normal(0, 1, size=self.obs_shape)
        cost = np.matmul(x.T, np.matmul(self.Q, x)) + np.matmul(action.T, np.matmul(self.R, action)) + 2 * \
               np.matmul(x.T, np.matmul(self.N, action))
        # print(cost)
        self.time += 1

        # Determine if the state is terminal
        done = 0
        exit_cond = 0
        if self.time >= self.horizon_length:
            done = 1
        if np.any(np.less(x, self.min_state)) or np.any(np.less(self.max_state, x)):
            # print(np.less(x, self.min_state))
            # print(np.less(self.max_state, x))
            exit_cond = 1

        # Convert values to outputs
        next_state = np.asarray(x_next.T)
        reward = -1 * float(cost)

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

        if evaluate and (self.eval_init is not None):
            self.state = np.matrix(self.eval_init).T
            self.time = 0
        else:
            self.state = np.matrix(
                [[np.random.uniform(self.min_init[i], self.max_init[i])] for i in range(len(self.state))])
            self.time = 0

        return


if __name__ == '__main__':
    sys = LQRRunner()
    # print(f'state: {sys.state}')
    # print(f'state.T: {sys.state.T}')
    sys.step(np.array([-1., -1., -1.]))
