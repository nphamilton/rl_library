"""
File:   convert_nn_to_matlab.py
Author: Nathaniel Hamilton

Description:    TODO

"""

import numpy as np
import torch
import scipy.io

from algorithms.ars.ars import *
from algorithms.ddpg.ddpg import *
from algorithms.ppo.ppo import *


def convert_nn_to_matlab(file_path, save_name, nn_type):
    """

    :param file_path:
    :param save_name:
    :param nn_type:
    :return:
    """
    if nn_type.lower() == 'ars':
        pass
    elif nn_type.lower() == 'ddpg':
        checkpoint = torch.load(file_path)

        # Store the saved models
        actor_state_dict = checkpoint['actor']

        weights = [actor_state_dict['linear1.weight'].numpy(), actor_state_dict['linear2.weight'].numpy(),
                   actor_state_dict['out.weight'].numpy()]

        biases = [actor_state_dict['linear1.bias'].numpy(), actor_state_dict['linear2.bias'].numpy(),
                  actor_state_dict['out.bias'].numpy()]

        # print(weights)

        scipy.io.savemat(save_name, mdict={'W': weights, 'b': biases})
    else:
        print('Invalid nn type given. Failed to convert.')


if __name__ == '__main__':
    convert_nn_to_matlab(
        file_path='/Users/nphamilton/rl_library/applications/mlv_project/cart_pole/ddpg/models/final_model.pth',
        save_name='test.mat',
        nn_type='ddpg')
