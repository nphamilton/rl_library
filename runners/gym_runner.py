"""
File:   gym_runner.py
Author: Nathaniel Hamilton

Description:

"""

import numpy as np
import gym

from runners.abstract_runner import Runner


class GymRunner(Runner):
    def __init__(self, env_name='CartPole-v0', render=True):
        """
        A runner works as an interface between the learning agent and the learning environment. Anything the agent wants to
        do in the environment should be run through a runner. Each environment should gets its own style of runner because
        every environment operates differently.

        Any method not included in this list should be preceded with __ to denote that is is unique to this specific
        runner. e.g.
            def __other_method(self): return

        Additionally, every runner should have the following class variables:
            obs_shape     (np.ndarray or np.array)
            action_shape  (np.array)
            is_discrete   (bool)
        """

        # Save input parameters
        self.render = render

        # Create the gym environment
        self.env = gym.make(env_name)

        # Declare the values for the required variables
        self.obs_shape = np.asarray(self.env.observation_space.shape)
        # print(self.obs_shape)
        if len(self.env.action_space.shape) == 0:
            self.is_discrete = True
            self.action_shape = np.asarray([self.env.action_space.n])
            print(self.action_shape)
        else:
            self.is_discrete = False
            self.action_shape = np.asarray(self.env.action_space.shape)
            # print(self.env.action_space.shape)

        self.state = np.zeros(self.obs_shape)

    def get_state(self):
        """
        This function should return the current state of the agent in the environment. No reward or status of done or
        exit will be provided, just the state.

        :input:
            None
        :output:
            return curr_state
        """
        return self.state

    def is_available(self):
        """
        This method checks to make sure the environment is still available. Some environments are able to discontinue
        without stopping the learning process.

        :input:
            None
        :output:
            return 0/1 (0 if unavailable, 1 if available)
        """
        return 1

    def step(self, action, render=False):
        """
        This function should execute a single step within the environment and return all necessary information
        including in the following order:
            1 next state/observation
            2 reward
            3 done (if the agent has reached a terminal state, this will be 1, otherwise 0)
            4 exit condition (if the agent has reached a fatal state, this will be 1, otherwise 0)

        :input:
            action  (np.array or int)
        :output:
            return next_state, reward, done, exit_cond
        """
        if render and self.render:
            self.env.render()

        next_state, reward, done, info = self.env.step(action)
        exit_cond = 0

        self.state = next_state

        return next_state, reward, done, exit_cond

    def stop(self):
        """
        This method should stop the agent from continuing any further actions. This could be a halt, or a termination of
        sorts. This should prevent the agent from hurting itself.

        :input:
            None
        :output:
            None
        """
        pass

    def reset(self, evaluate=False):
        """
        This function should reset the environment. In the case of a simulation, this should take the agent back to a
        safe starting point. In a real-world system, this might involve halting until a resume signal has been sent
        allowing the user to move the agent to a safe starting location.

        :input:
            evaluate    (bool) When this is true, the environment should reset to set starting point for consistent
                                evaluation
        :output:
            Nothing is returned from this function.
        """
        self.state = self.env.reset()

        return


