"""
File:   avg_buck_converter.py
Author: Nathaniel Hamilton

Description:    This runner simulates a Linear Quadratic Regression (LQR) system.


"""
import numpy as np
import copy
import matplotlib.pyplot as plt

from runners.abstract_runner import Runner


class AvgBuckConverter(Runner):
    def __init__(self, capacitor_value=4.4e-6, capacitor_tolerance=0.2, inductor_value=5.0e-5, inductor_tolerance=0.2,
                load_avg=6.0, load_range=[4.0, 8.0],
                 sample_time=0.00001,
                 switching_frequency=10e-3, source_voltage=10.0, reference_voltage=9.0, desired_voltage=6.0,
                 max_action=np.array([0.95]), min_action=np.array([0.05]),
                 max_state=np.array([1000., 1000.]), min_state=None, scale=1,
                 max_init_state=np.array([3., 3.]), min_init_state=None,
                 evaluation_init=np.array([0., 0.])):
        """

        :param state_matrix:        (np.matrix) nxn matrix
        :param input_matrix:        (np.matrix) nxm matrix
        :param state_cost:          (np.array)   n length array
        :param input_cost:          (np.array)   m length array
        :param cross_term:
        :param horizon_length:
        :param evaluation_init:
        """
        # Save the inputs
        self.capacitor_nom = capacitor_value
        self.capacitor_range = capacitor_value * np.asarray([(1 - capacitor_tolerance), (1 + capacitor_tolerance)])
        self.inductor_nom = inductor_value
        self.inductor_range = inductor_value * np.asarray([(1 - inductor_tolerance), (1 + inductor_tolerance)])
        self.load_nom = load_avg
        self.load_range = load_range
        self.dt = sample_time
        self.Vs = source_voltage
        self.Vref = reference_voltage
        self.Vdes = desired_voltage
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

        # Establish the required variables for learning algorithms to access
        self.obs_shape = np.array([4])
        # print(f'lqr.obs_shape: {self.obs_shape[0]}')
        self.action_shape = np.array([1])
        # print(f'lqr.action_shape: {self.action_shape[0]}')
        self.is_discrete = False

        # Initialize variables
        self.C = capacitor_value
        self.L = inductor_value
        self.R = load_avg
        self.max_time = 0.00001 / (self.R * self.C)
        self.state = np.zeros_like(self.max_state)
        self.scale_mult = (self.max_action - self.min_action) / 2.0
        self.scale_add = (self.max_action - self.min_action) / 2.0 + self.min_action
        self.time = 0
        self.stable_count = 0

    def get_state(self):
        """
        This function should return the current state of the agent in the environment. No reward or status of done or
        exit will be provided, just the state.

        """
        i = self.state[0]
        v = self.state[1]
        observation = np.asarray([(self.Vref-v), self.Vref, v, i])
        return observation

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
            action = np.multiply(action, self.scale_mult) + self.scale_add
        elif self.scale == 0:
            action = np.minimum(np.maximum(action, self.min_action), self.max_action)
        else:
            raise NotImplementedError

        # Compute the next state
        R = self.R  # TODO: add randomness to the load that changes at each step
        A = np.array([[0.0, -1.0 / self.L], [1.0 / self.C, -1.0 / (R * self.C)]])  # state x=[i,v]
        # print(f'A = {A}')
        x_next = copy.copy(x)
        time_range = np.arange(0, self.dt, 0.0000001)
        start_time = self.time
        for i in time_range:
            Vref = self.Vref  # + self.amp*np.sin(self.freq*(start_time + i))
            D = Vref / self.Vs
            B = D * np.asarray([self.Vs / self.L, 0.0])
            dx = np.dot(A, x_next) + np.multiply(B, action)
            # print(dx*self.dt)
            x_next = x_next + dx*0.0000001
        # print(f'next = {x_next}')

        # Store and convert values
        self.state = x_next
        next_obs = self.get_state()
        self.time += self.dt

        # Compute the reward
        # reward = -1 * next_obs[0]**2  # -(Vref - v)^2
        reward = -1 * (self.Vdes - next_obs[2])**2  # -(Vdes - Vout)^2

        # Determine if the state is terminal
        done = 0
        exit_cond = 0
        if np.sum(x - x_next) == 0.0:
            self.stable_count += 1
        else:
            self.stable_count = 0
        if self.stable_count >= 10:
            done = 1
        if self.time >= self.max_time:
            exit_cond = 1
        if np.any(np.less(x_next, self.min_state)) or np.any(np.less(self.max_state, x_next)):
            # print(np.less(x, self.min_state))
            # print(np.less(self.max_state, x))
            exit_cond = 1
            reward = -100.

        return next_obs, reward, done, exit_cond

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
            self.C = self.capacitor_nom
            self.L = self.inductor_nom
            self.R = self.load_nom
            self.state = self.eval_init
            self.time = 0
            self.stable_count = 0
        else:
            self.C = np.random.uniform(self.capacitor_range[0], self.capacitor_range[1], 1)[0]
            self.L = np.random.uniform(self.inductor_range[0], self.inductor_range[1], 1)[0]
            self.R = np.random.uniform(self.load_range[0], self.load_range[1], 1)[0]
            self.state = np.asarray(
                [np.random.uniform(self.min_init[i], self.max_init[i]) for i in range(len(self.state))])
            self.time = 0
            self.stable_count = 0

        return


if __name__ == '__main__':
    sys = AvgBuckConverter()
    # print(f'state: {sys.state}')
    # print(f'state.T: {sys.state.T}')
    sys.reset(True)
    done = 0
    exit_cond = 0
    currents = [0.]
    voltages = [0.]
    reward = 0
    while done == 0 and exit_cond == 0:
        n, r, done, exit_cond = sys.step(np.array([1.]))
        reward += r
        currents.append(n[3])
        voltages.append(n[2])
    print(f'Done: {done}')
    print(f'Exit: {exit_cond}')
    print(f'Reward: {reward}')
    fig, ax = plt.subplots()
    ax.plot(currents, voltages)
    ax.set(xlim=[0.0, 2.5], ylim=[0.0, 8.0])
    plt.show()
    # print(voltages)
    # print(n)
    # n, _, _, _ = sys.step(np.array([0., 0.]))
    # print(n)
