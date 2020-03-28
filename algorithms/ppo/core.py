"""
File:   core.py
Author: Nathaniel Hamilton

Description: This class implements the associated classes for the Proximal Policy Optimization algorithm written
             about in https://arxiv.org/abs/1707.06347

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_log_probability(x, mu, std_devs):
    """
    The log likelihood of a multivariate Gaussian is computed using the following formula:
        ln(L) = -0.5(ln(det(Var) + (x-mu)'*Var^(-1)*(x-mu) + kln(2*pi))
        ln(L) = -0.5(2*sum(ln(std_i)) + sum(((x_i - mu_i)/std_i)^2) + ln(2*pi))

    :param x:           (ndarray)
    :param mu:          (FloatTensor)
    :param std_devs:    (FloatTensor)

    :return log_prob:   (float)
    """
    # Make sure nothing is attached
    mu = mu.detach()
    std_devs = std_devs.detach()

    # Compute first term
    logstds = torch.log(std_devs)
    a = 2 * torch.sum(logstds).numpy()

    # Compute second term
    x = torch.FloatTensor(x)
    b = torch.sum(torch.pow(((x - mu) / std_devs), 2)).numpy()

    # Compute third term
    c = np.log(2*np.pi)

    # Combine the terms
    log_prob = -0.5 * (a + b + c)

    return log_prob


def init_weights(m):
    """
    Function for initializing layers with orthogonal weights.

    :param m: (tensor)  the layer to be orthogonally weighted.
    """

    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)

    return


########################################################################################################################
class ContinuousActor(nn.Module):
    def __init__(self, num_inputs=4, hidden_size1=64, hidden_size2=64, num_actions=2):
        super(ContinuousActor, self).__init__()
        """
        This Neural Network architecture creates an actor, which provides the control output. The architecture is 
        derived from the original PPO paper.

        :param num_inputs:  (int)   The desired size of the input layer. Should be the same size as the number of 
                                        inputs to the NN. Default=4
        :param hidden_size1:(int)   The desired size of the first hidden layer. Default=400
        :param hidden_size2:(int)   The desired size of the second hidden layer. Default=300
        :param num_actions: (int)   The desired size of the output layer. Should be the same size as the number of 
                                        outputs from the NN. Default=2
        """

        # The first layer
        self.linear1 = nn.Linear(num_inputs, hidden_size1)

        # The second layer
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)

        # The output layer
        self.out = nn.Linear(hidden_size2, num_actions)

        # Initialize according to method described in PPO paper
        self.initialize_orthogonal()

        # Compute the standard deviation variable
        self.std = torch.exp(torch.as_tensor(-0.5 * np.ones(num_actions, dtype=np.float32)))

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

    def forward(self, state, action=None):
        """
        This function performs a forward pass through the network.

        :param state:  (tensor)   The input state the NN uses to compute an output.
        :param action: (np.array) ???
        :return mu:    (tensor)   The output of the NN, which is the action to be taken.
        """

        # Pass through layer 1
        x = self.linear1(state)
        x = torch.tanh(x)

        # Pass through layer 2
        x = self.linear2(x)
        x = torch.tanh(x)

        # Pass through the output layer
        x = self.out(x)
        mu = torch.tanh(x)

        # Compute the log probability of taking the input action
        logp_a = None
        if action is not None:
            logp_a = calculate_log_probability(action, mu, self.std)

        # Return the result
        return mu, self.std, logp_a  # TODO modify if necessary to make discrete option match

# TODO discrete actor


########################################################################################################################
class PPOCritic(nn.Module):
    def __init__(self, num_inputs=4, hidden_size1=64, hidden_size2=64):
        super(PPOCritic, self).__init__()
        """
        This Neural Network architecture creates a full critic, which estimates the value of a state-action pair. The 
        architecture is derived from the original PPO paper.

        :param num_inputs:  (int)   The desired size of the input layer. Should be the same size as the number of 
                                        inputs to the NN. Default=4
        :param hidden_size1:(int)   The desired size of the first hidden layer. Default=400
        :param hidden_size2:(int)   The desired size of the second hidden layer. Default=300
        :param num_actions: (int)   The number of actions the network will receive. Should be the same size as the 
                                        number of outputs from the ActorNN. Default=2
        :param final_bias:  (float) The final layers' weight and bias range for uniform distribution. Default=3e-3
        """

        # The first layer
        self.linear1 = nn.Linear(num_inputs, hidden_size1)

        # The second layer
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)

        # The output layer
        self.out = nn.Linear(hidden_size2, 1)

        # Initialize layers according to PPO paper
        self.initialize_orthogonal()

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

    def forward(self, state):
        """
        This function performs a forward pass through the network.

        :param state:   (tensor)    The input state/observation for the NN to compute the state value.
        :return v:      (tensor)    An estimated value of the input state/observation.
        """

        # Pass through layer 1
        x = self.linear1(state)
        x = torch.tanh(x)

        # Pass through layer 2
        x = self.linear2(x)
        x = torch.tanh(x)

        # Pass through the output layer
        v = self.out(x)

        # Return the result
        return v
