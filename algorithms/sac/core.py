"""
File:   core.py
Author: Nathaniel Hamilton

Description: This class implements the associated classes for the Soft Actor-Critic algorithm written
             about in https://arxiv.org/abs/1801.01290

Disclaimer: Some of this code is copied from OpenAI's SpinningUp implementation at
            https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def soft_update(target, source, tau):
    """
    This function performs a soft update between two neural networks. The target's parameters are modified towards the
    source's by a factor of tau.

    :param target:  (torch.nn)  The target neural network that is being updated
    :param source:  (torch.nn)  The source neural network the target is updated towards
    :param tau:     (float)     Scale value between 0 and 1
    :return:
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    return


def hard_update(target, source):
    """
    This function performs a hard update between two neural networks. In a hard update, the parameters of the source
    are copied to the target.

    :param target:  (torch.nn)  The target neural network that is being updated
    :param source:  (torch.nn)  The source neural network the target is updated towards
    :return:
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

    return


def init_weights(m):
    """
    Function for initializing layers with orthogonal weights.

    :param m: (tensor)  the layer to be orthogonally weighted.
    """

    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)

    return


def fan_in_uniform_init(m):
    """
    Initializes the given layer uniformly, as prescribed in the DDPG paper

    :param m: (tensor) The layer to be uniformly weighted.
    """

    weight_range = 1.0 / np.sqrt(m.size(-1))
    nn.init.uniform_(m, -weight_range, weight_range)

    return


LOG_STD_MAX = 2
LOG_STD_MIN = -20


########################################################################################################################
class SACActor(nn.Module):
    def __init__(self, num_inputs=4, hidden_size1=256, hidden_size2=256, num_actions=2):
        super(SACActor, self).__init__()
        """
        This Neural Network architecture creates a full actor, which provides the control output. The architecture is 
        derived from the original SAC paper.

        :param num_inputs:  (int)   The desired size of the input layer. Should be the same size as the number of 
                                        inputs to the NN. Default=4
        :param hidden_size1:(int)   The desired size of the first hidden layer. Default=256
        :param hidden_size2:(int)   The desired size of the second hidden layer. Default=256
        :param num_actions: (int)   The desired size of the output layer. Should be the same size as the number of 
                                        outputs from the NN. Default=2
        :param final_bias:  (float) The final layers' weight and bias range for uniform distribution. Default=3e-3
        """

        # The first layer
        self.linear1 = nn.Linear(num_inputs, hidden_size1)

        # The second layer
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)

        # The output layers
        self.mu_out = nn.Linear(hidden_size2, num_actions)
        self.log_std_out = nn.Linear(hidden_size2, num_actions)

    def initialize_orthogonal(self):
        """
        This function initializes the weights of the network according to the method described in "Exact solutions to
        the nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al. (2013)
        """

        # Initialize linear1
        self.linear1.apply(init_weights)

        # Initialize linear2
        self.linear2.apply(init_weights)

        # Initialize output layer
        self.mu_out.apply(init_weights)
        self.log_std_out.apply(init_weights)

        return

    def forward(self, state, deterministic=False, with_logp=True):
        """
        This function performs a forward pass through the network.

        :param state:           (tensor)    The input state the NN uses to compute an output.
        :param deterministic:   (bool)      Flag indicating whether or not the returned action should be deterministic.
        :param with_logp:       (bool)      Flag indicating whether or not the logp should be calculated.
        :return pi_action:      (tensor)    The output of the NN, which is the action to be taken.
        :return logp_pi:        (tensor)    The log probability of the output action.
        """

        # Pass through layer 1
        x = self.linear1(state)
        x = F.relu(x)

        # Pass through layer 2
        x = self.linear2(x)
        x = F.relu(x)

        # Pass through the output mu layer
        mu = self.mu_out(x)

        # Pass through the output log_std layer
        log_std = self.log_std_out(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logp:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        return pi_action, logp_pi


########################################################################################################################
class SACCritic(nn.Module):
    def __init__(self, num_inputs=4, hidden_size1=256, hidden_size2=256, num_actions=2):
        super(SACCritic, self).__init__()
        """
        This Neural Network architecture creates a full critic, which estimates the value of a state-action pair. The 
        architecture is derived from the original SAC paper.

        :param num_inputs:  (int)   The desired size of the input layer. Should be the same size as the number of 
                                        inputs to the NN. Default=4
        :param hidden_size1:(int)   The desired size of the first hidden layer. Default=256
        :param hidden_size2:(int)   The desired size of the second hidden layer. Default=256
        """

        # The first layer
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size1)

        # The second layer
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)

        # The output layer
        self.out = nn.Linear(hidden_size2, 1)

    def initialize_orthogonal(self):
        """
        This function initializes the weights of the network according to the method described in "Exact solutions to
        the nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al. (2013)
        """

        # Initialize linear1
        self.linear1.apply(init_weights)

        # Initialize linear2
        self.linear2.apply(init_weights)

        # Initialize output layer
        self.out.apply(init_weights)

        return

    def forward(self, state, action):
        """
        This function performs a forward pass through the network.

        :param state:   (tensor)    The input state for the NN to compute the state-action value.
        :param action:  (tensor)    The input action for the NN to compute the state-action value.
        :return q:      (tensor)    An estimated Q-value of the input state-action pair.
        """

        # Concatenate the first layer output with the action
        x = torch.cat((state, action), 1)

        # Pass through layer 1
        x = self.linear1(x)
        x = F.relu(x)

        # Pass through layer 2
        x = self.linear2(x)
        x = F.relu(x)

        # Pass through the output layer
        q = self.out(x)
        q = F.relu(q)

        # Return the result
        return q
