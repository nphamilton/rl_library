"""
File:   abstract_algorithm.py
Author: Nathaniel Hamilton

Description:

Usage:          This class should be inherited by each implemented algorithm class in order to enforce consistency.

"""

from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    An algorithm is how an agent learns to perform a given task.

    Any method not included in this list should be preceded with __ to denote that is is unique to the specific
    algorithm. e.g.
        def __other_method(self): return
    """

    @abstractmethod
    def evaluate_model(self, evaluation_length):
        """
        This method performs an evaluation of the model. The evaluation lasts for the specified number of executed
        steps. Multiple evaluations should be used to account for variations in performance.

        :input:
            evaluation_length
        :output:
            return reward_sum, steps, done, exit_cond
        """
        pass

    @abstractmethod
    def load_model(self, load_path):
        """
        This method loads a model. The loaded model can be a previously learned policy or an initializing policy used
        for consistency.

        :input:
            load_path
        :output:
            None
        """
        pass

    @abstractmethod
    def save_model(self, save_path):
        """
        This method saves the current model learned by the agent.

        :input:
            save_path
        :output:
            None
        """
        pass

    @abstractmethod
    def train_model(self):
        """
        This method trains the agent to learn an optimal model in the manner specified by the learning algorithm.

        :input:
            None (all parameters and hyperparameters should be specified upon initialization
        :output:
            return final_policy_reward_sum, execution_time
        """
        pass

    @abstractmethod
    def update_model(self):
        """
        This method executes the model update according to the specific learning algorithm.

        Inputs and outputs for this method depend on the learning algorithm.
        """
        pass
