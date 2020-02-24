"""
File:   f1_10.py
Author: Nathaniel Hamilton

Description: This class implements a runner for the F1/10th racing simulator. The goal of an agent using this runner is
             to complete laps as quickly as possible.

Usage:       Import the entire class file to instantiate and use this runner.

"""

from runners.abstract_runner import Runner


class F110Runner(Runner):
    def __init__(self):
        """
        TODO: describe this runner
        """
        # TODO: figure out all the initializations

    def get_state(self):
        """
        This function returns the current state of the agent in the environment.

        :input:
            None
        :output:
            curr_state
        """
        # TODO: write this

        return

    def step(self):
        """
        This function should execute a single step within the environment and return all necessary information
        including in the following order:
            1 next state/observation
            2 reward
            3 done (if the agent has reached a terminal state, this will be 1, otherwise 0)
            4 exit condition (if the agent has reached a fatal state, this will be 1, otherwise 0)

        :input:
            action
        :output:
            return next_state, reward, done, exit
        """
        # TODO: write this

        return

    def reset(self):
        """
        This function should reset the environment. In the case of a simulation, this should take the agent back to a
        safe starting point. In a real-world system, this might involve halting until a resume signal has been sent
        allowing the user to move the agent to a safe starting location.

        :output:
            Nothing is returned from this function.
        """
        # TODO: write this

        return

