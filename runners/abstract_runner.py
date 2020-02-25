"""
File:   abstract_runner.py
Author: Nathaniel Hamilton

Description:    A runner works as an interface between the learning agent and the learning environment. Anything the
                agent wants to do in the environment should be run through a runner. Each environment should gets its
                own style of runner because every environment operates differently.

Usage:          This class should be inherited by each implemented runner class in order to enforce consistency.

"""

from abc import ABC, abstractmethod


class Runner(ABC):
    """
    A runner works as an interface between the learning agent and the learning environment. Anything the agent wants to
    do in the environment should be run through a runner. Each environment should gets its own style of runner because
    every environment operates differently.

    Any method not included in this list should be preceded with __ to denote that is is unique to this specific
    runner. e.g.
        def __other_method(self): return
    """

    @abstractmethod
    def get_state(self):
        """
        This function should return the current state of the agent in the environment. No reward or status of done or
        exit will be provided, just the state.

        :input:
            None
        :output:
            return curr_state
        """
        pass

    @abstractmethod
    def is_available(self):
        """
        This method checks to make sure the environment is still available. Some environments are able to discontinue
        without stopping the learning process.

        :input:
            None
        :output:
            return 0/1 (0 if unavailable, 1 if available)
        """
        pass

    @abstractmethod
    def step(self, action):
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
        pass

    @abstractmethod
    def stop(self):
        """
        This method should stop the agent from continuing any further actions. This could be a halt, or a termination of
        sorts. This should prevent the agent from hurting itself.

        :input:
            None
        :output:
            None
        """

    @abstractmethod
    def reset(self):
        """
        This function should reset the environment. In the case of a simulation, this should take the agent back to a
        safe starting point. In a real-world system, this might involve halting until a resume signal has been sent
        allowing the user to move the agent to a safe starting location.

        :input:
            None
        :output:
            Nothing is returned from this function.
        """
        pass


