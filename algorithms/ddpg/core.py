"""
File:   core.py
Author: Nathaniel Hamilton

Description: This class implements the associated classes for the Deep Deterministic Policy Gradient algorithm written
             about in https://arxiv.org/abs/1509.02971

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


########################################################################################################################
class DDPGActor(nn.Module):
    def __init__(self, num_inputs=4, hidden_size1=400, hidden_size2=300, num_actions=2, final_bias=3e-3):
        super(DDPGActor, self).__init__()
        """
        This Neural Network architecture creates a full actor, which provides the control output. The architecture is 
        derived from the original DDPG paper.

        :param num_inputs:  (int)   The desired size of the input layer. Should be the same size as the number of 
                                        inputs to the NN. Default=4
        :param hidden_size1:(int)   The desired size of the first hidden layer. Default=400
        :param hidden_size2:(int)   The desired size of the second hidden layer. Default=300
        :param num_actions: (int)   The desired size of the output layer. Should be the same size as the number of 
                                        outputs from the NN. Default=2
        :param final_bias:  (float) The final layers' weight and bias range for uniform distribution. Default=3e-3
        """

        # The first layer
        self.linear1 = nn.Linear(num_inputs, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)

        # The second layer
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)

        # The output layer
        self.out = nn.Linear(hidden_size2, num_actions)

        # Initialize layers according to DDPG paper
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.out.weight, -final_bias, final_bias)

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

        :param state: (tensor)   The input state the NN uses to compute an output.
        :return mu:   (tensor)   The output of the NN, which is the action to be taken.
        """

        # Pass through layer 1
        x = self.linear1(state)
        x = F.relu(x)
        x = self.ln1(x)

        # Pass through layer 2
        x = self.linear2(x)
        x = F.relu(x)
        x = self.ln2(x)

        # Pass through the output layer
        x = self.out(x)

        # Output is scaled using tanh as prescribed in DDPG paper
        mu = torch.tanh(x)

        # Return the result
        return mu


########################################################################################################################
class DDPGCritic(nn.Module):
    def __init__(self, num_inputs=4, hidden_size1=400, hidden_size2=300, num_actions=2, final_bias=3e-3):
        super(DDPGCritic, self).__init__()
        """
        This Neural Network architecture creates a full critic, which estimates the value of a state-action pair. The 
        architecture is derived from the original DDPG paper.

        :param num_inputs:  (int)   The desired size of the input layer. Should be the same size as the number of 
                                        inputs to the NN. Default=4
        :param hidden_size1:(int)   The desired size of the first hidden layer. Default=400
        :param hidden_size2:(int)   The desired size of the second hidden layer. Default=300
        :param num_actions: (int)   The number of actions the network will recieve. Should be the same size as the 
                                        number of outputs from the ActorNN. Default=2
        :param final_bias:  (float) The final layers' weight and bias range for uniform distribution. Default=3e-3
        """

        # The first layer
        self.linear1 = nn.Linear(num_inputs, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)

        # The second layer
        self.linear2 = nn.Linear(hidden_size1 + num_actions, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)

        # The output layer
        self.out = nn.Linear(hidden_size2, num_actions)

        # Initialize layers according to DDPG paper
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.out.weight, -final_bias, final_bias)

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

        # Pass through layer 1
        x = self.linear1(state)
        x = F.relu(x)
        x = self.ln1(x)

        # Concatenate the first layer output with the action
        x = torch.cat((x, action), 1)

        # Pass through layer 2
        x = self.linear2(x)
        x = F.relu(x)
        x = self.ln2(x)

        # Pass through the output layer
        q = self.out(x)

        # Return the result
        return q


########################################################################################################################
class VerivitalActor(nn.Module):
    def __init__(self, num_inputs=4, hidden_size1=400, hidden_size2=300, num_actions=2, final_bias=3e-3):
        super(VerivitalActor, self).__init__()
        """
        This Neural Network architecture creates a full actor, which provides the control output. The architecture is 
        derived from the original DDPG paper.

        :param num_inputs:  (int)   The desired size of the input layer. Should be the same size as the number of 
                                        inputs to the NN. Default=4
        :param hidden_size1:(int)   The desired size of the first hidden layer. Default=400
        :param hidden_size2:(int)   The desired size of the second hidden layer. Default=300
        :param num_actions: (int)   The desired size of the output layer. Should be the same size as the number of 
                                        outputs from the NN. Default=2
        :param final_bias:  (float) The final layers' weight and bias range for uniform distribution. Default=3e-3
        """

        # The first layer
        self.linear1 = nn.Linear(num_inputs, hidden_size1)
        # self.ln1 = nn.LayerNorm(hidden_size1)

        # The second layer
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        # self.ln2 = nn.LayerNorm(hidden_size2)

        # The output layer
        self.out = nn.Linear(hidden_size2, num_actions)

        # Initialize layers according to DDPG paper
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.out.weight, -final_bias, final_bias)

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

        :param state: (tensor)   The input state the NN uses to compute an output.
        :return mu:   (tensor)   The output of the NN, which is the action to be taken.
        """

        # Pass through layer 1
        x = self.linear1(state)
        x = F.relu(x)
        # x = self.ln1(x)

        # Pass through layer 2
        x = self.linear2(x)
        x = F.relu(x)
        # x = self.ln2(x)

        # Pass through the output layer
        x = self.out(x)

        # Output is scaled using hardtanh, which is a linearized version of the tanh function
        mu = F.hardtanh(x)

        # Return the result
        return mu