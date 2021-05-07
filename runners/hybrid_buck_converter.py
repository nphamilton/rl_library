"""
File:   avg_buck_converter.py
Author: Nathaniel Hamilton

Description:    This runner simulates a Linear Quadratic Regression (LQR) system.


"""
import numpy as np
from scipy import signal
import copy
import matplotlib.pyplot as plt

from runners.abstract_runner import Runner


class HybridBuckConverter(Runner):
    def __init__(self, capacitor_value=4.4e-6, capacitor_tolerance=0.2, inductor_value=5.0e-5, inductor_tolerance=0.2,
                 load_avg=6.0, load_range=np.asarray([4.0, 8.0]),
                 sample_time=0.00001,
                 switching_frequency=10e3, source_voltage=10.0, reference_voltage=9.0, desired_voltage=6.0,
                 switching_loss=0.0, inductor_loss=0.0,
                 max_action=np.array([0.95]), min_action=np.array([0.05]),
                 max_state=np.array([1000., 1000.]), min_state=np.array([-100., -100.]), scale=1,
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
        # Convert the inputs
        self.C = capacitor_value
        self.L = inductor_value
        self.R = load_avg
        self.dt = sample_time
        self.switch_freq = switching_frequency
        self.Vs = source_voltage
        self.Vref = reference_voltage
        self.Vdes = desired_voltage
        self.rs = switching_loss
        self.rL = inductor_loss

        # Save the parameters
        self.A = np.array([[0.0, -1.0 / self.L], [1.0 / self.C, -1.0 / (self.R * self.C)]])  # state x=[i,v]
        # print(f'A = {self.A}')
        self.max_time = 0.1 / (self.R * self.C)
        self.sim_dt = 0.0000001
        self.eval_init = evaluation_init
        self.scale = scale

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

        self.min_state = min_state
        self.max_state = max_state

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
        self.state = np.zeros_like(self.max_state)
        self.scale_mult = (self.max_action - self.min_action) / 2.0
        self.scale_add = (self.max_action - self.min_action) / 2.0 + self.min_action
        self.time = 0
        self.stable_count = 0

        # Generate the sawtooth function that is used for determining if the switch is open or closed
        # TODO: explain how this works
        self.sawtooth_period = 1. / self.switch_freq
        # sawtooth_t = np.arange(0., 1., self.sim_dt / self.sawtooth_period)
        steps_in_2_periods = round((2 * self.sawtooth_period) / self.sim_dt)
        # print(steps_in_2_periods)
        sawtooth_t = np.linspace(0., (steps_in_2_periods * self.sim_dt), steps_in_2_periods)
        self.sawtooth = (signal.sawtooth(2 * np.pi * self.switch_freq * sawtooth_t) + 1.) / 2.
        # print(self.sawtooth)
        self.sawtooth_len = len(self.sawtooth)
        # print(self.sawtooth_len)
        self.last_sawtooth_val = 0.0
        self.mode = 0

    def get_state(self):
        """
        This function should return the current state of the agent in the environment. No reward or status of done or
        exit will be provided, just the state.

        """
        i = self.state[0]
        v = self.state[1]
        observation = np.asarray([(self.Vref - v), self.Vref, v, i])
        return observation

    def is_available(self):
        """
        This environment is always available, so returns True.
        """
        return True

    def __charging_dynamics(self, x_start, load, t_start, threshold, t_max):
        """
        In this function, the state is changing according to the charging dynamics. The system is charging when the
        switch is closed, i.e. sawtooth_pwm - threshold <= 0. This might also be referred to as "closed". The transition
        matrices are defined as:
            A = [-1*(rs+rL)/L, -1/L;
                 1/C,          -1/(R*C)]
            B = [Vs/L, 0]'
        where x' = Ax + B
        :return:
        """
        A = np.array([[-1 * (self.rs + self.rL) / self.L, -1.0 / self.L], [1.0 / self.C, -1.0 / (load * self.C)]])
        B = np.array([self.Vs / self.L, 0.0])
        x_next = copy.copy(x_start)
        t = t_start
        pos = int((t / self.sim_dt) % self.sawtooth_len)
        # print("charge pos: " + str(pos))
        while t <= t_max and self.sawtooth[pos] <= threshold:
            dx = np.dot(A, x_next) + B
            x_next = x_next + dx * self.sim_dt
            t += self.sim_dt
            pos += 1
            if pos >= self.sawtooth_len:
                pos = pos % self.sawtooth_len
        mode_next = 1  # The next mode after charging is always discharging
        time_covered = t - t_start
        # print("pos: " + str(pos) + " u: " + str(threshold) + " t: " + str(time_covered))
        return x_next, mode_next, time_covered

    def __discharging_dynamics(self, x_start, load, t_start, threshold, t_max):
        """
        In this function, the state is changing according to the discharging dynamics. The system is discharging when
        the switch is open, i.e. sawtooth_pwm - threshold > 0. This might also be referred to as "open". The transition
        matrices are defined as:
            A = [-rL/L, -1/L;
                 1/C,          -1/(R*C)]
        where x' = Ax
        :return:
        """
        # Make sure the current is positive, jumping to discontinuous dynamics if it isn't
        if x_start[0] <= 0.0:
            # print('Not discharging. Oops!')
            return x_start, 2, t_start

        A = np.array([[-1 * self.rL / self.L, -1.0 / self.L], [1.0 / self.C, -1.0 / (load * self.C)]])
        x_next = copy.copy(x_start)
        mode_next = 0  # The default next mode after discharging is to charge again
        t = t_start
        pos = int((t / self.sim_dt) % self.sawtooth_len)
        # print("discharge pos: " + str(pos))
        # print(f't <= t_max: {t <= t_max}, sawtooth[pos] > threshold: {self.sawtooth[pos] > threshold}')
        while t <= t_max and self.sawtooth[pos] > threshold:
            dx = np.dot(A, x_next)
            x_next_temp = x_next + dx * self.sim_dt
            t += self.sim_dt
            pos += 1
            if pos >= self.sawtooth_len:
                pos = pos % self.sawtooth_len

            # If the current is <= 0, the next mode is discontinuous
            if x_next_temp[0] <= 0.0:
                # print(f'undoing discharge step: {x_next_temp[0]}')
                mode_next = 2
                t -= self.sim_dt
                break
            else:
                x_next = x_next_temp
        # print(f't <= t_max: {t <= t_max}, sawtooth[pos] > threshold: {self.sawtooth[pos] > threshold}')
        time_covered = t - t_start

        return x_next, mode_next, time_covered

    def __discontinuous_dynamics(self, x_start, load, t_start, threshold, t_max):
        """
        In this function, the state is changing according to the discontinuous dynamics. The transition matrices are
        defined as:
            A = [0, 0;
                 0, -1/(R*C)]
        where x' = Ax
        :return:
        """
        A = np.array([[0.0, 0.0], [0.0, -1.0 / (load * self.C)]])
        x_next = copy.copy(x_start)
        t = t_start
        pos = int((t / self.sim_dt) % self.sawtooth_len)
        while t <= t_max:
            dx = np.dot(A, x_next)
            x_next = x_next + dx * self.sim_dt
            t += self.sim_dt
            pos += 1
            if pos >= self.sawtooth_len:
                pos = pos % self.sawtooth_len

            if (x_next[0] >= 0.0) and (self.sawtooth[pos] > threshold):
                # If the current is above 0 and the voltage is below the reference point, then the mode switches back
                # to charging
                break
        time_covered = t - t_start
        mode_next = 0  # The next mode after discontinuous is always charging

        return x_next, mode_next, time_covered

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

        # Randomly select the load for the given step
        load = self.R  # TODO: add randomness to the load that changes at each step

        # Adjust the action
        if self.scale == 1:
            # Scale the action
            action = np.multiply(action, self.scale_mult) + self.scale_add
        elif self.scale == 0:
            action = np.minimum(np.maximum(action, self.min_action), self.max_action)
        else:
            raise NotImplementedError

        # Compute the next state
        x_next = copy.copy(x)
        t = self.time
        t_fin = t + self.dt
        mode = self.mode
        while t <= t_fin:
            # print("time: " + str(t))
            if mode == 0:  # If the mode is 'charging'
                # print("charging")
                x_next, mode_next, time_covered = self.__charging_dynamics(x_next, load, t, action, t_fin)
            elif mode == 1:  # If the mode is 'discharging'
                # print("discharging")
                x_next, mode_next, time_covered = self.__discharging_dynamics(x_next, load, t, action, t_fin)
            elif mode == 2:  # If the mode is 'discontinuous'
                # print("discontinuous")
                x_next, mode_next, time_covered = self.__discontinuous_dynamics(x_next, load, t, action, t_fin)
            else:
                print('This mode does not exist and the code is broken')
                time_covered = 0.
                mode_next = 7
            # print("time_covered " + str(time_covered))
            t += time_covered
            # print("next time: " + str(t))
            # Ensure the state never exceeds the physical bounds
            # x_next = np.minimum(np.maximum(x_next, self.min_state), self.max_state)
            mode = mode_next

        # Store and convert values
        self.state = x_next
        # print(x_next)
        self.mode = mode
        next_obs = self.get_state()
        self.time += self.dt

        # Compute the reward
        # reward = -1 * (self.Vdes - self.state[1]) ** 2  # -(Vdes - Vout)^2
        reward = -1 * abs(self.Vdes - self.state[1])  # -|Vdes - Vout|

        # Determine if the state is terminal
        done = 0
        exit_cond = 0
        if np.sum(x - x_next) <= 0.1 and abs(self.state[1] - self.Vdes) <= 0.1:
            self.stable_count += 1
            # print("stable " + str(self.stable_count))
        else:
            self.stable_count = 0
        if self.stable_count >= 10:
            done = 1
        if self.time >= self.max_time:
            exit_cond = 1
        if np.any(np.less(x_next, self.min_state)) or np.any(np.less(self.max_state, x_next)):
            print(f'observed state is below the min: {np.less(x_next, self.min_state)}')
            print(f'observed state is above the max: {np.less(self.max_state, x_next)}')
            print(f'state: {x_next}, step count: {self.time / self.dt}')
            exit_cond = 1
            reward = -100.

        if render:
            print(f'step count: {round(self.time / self.dt)}, state: {x_next}, action: {action}, reward: {reward}')
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
            self.time = 0.0
            self.stable_count = 0
            self.mode = 0
        else:
            # self.C = np.random.uniform(self.capacitor_range[0], self.capacitor_range[1], 1)[0]
            # self.L = np.random.uniform(self.inductor_range[0], self.inductor_range[1], 1)[0]
            # self.R = np.random.uniform(self.load_range[0], self.load_range[1], 1)[0]
            self.C = self.capacitor_nom
            self.L = self.inductor_nom
            self.R = self.load_nom
            self.state = np.asarray(
                [np.random.uniform(self.min_init[i], self.max_init[i]) for i in range(len(self.state))])
            # self.state = self.eval_init
            self.time = 0.0
            self.stable_count = 0
            self.mode = 0

        return


if __name__ == '__main__':
    sys = HybridBuckConverter()
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
