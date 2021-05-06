"""
TODO header
"""
import numpy as np
from scipy.io import savemat
from algorithms.ddpg.ddpg import *
from runners.hybrid_buck_converter import *

if __name__ == '__main__':
    # Declare all the variables
    capacitance = 2.2e-3  # C
    inductance = 2.65e-3  # L
    load = 10.0  # R
    source_voltage = 100.0  # v_s
    reference_voltage = 48.0  # v_ref
    sample_time = 0.001  # 1/f_s
    switch_period = 1e4  # 1/f_sw
    rollout_length = 2000

    # Create the runner
    path = './data/hybrid_dynamics'
    runner = HybridBuckConverter(capacitor_value=capacitance, capacitor_tolerance=0.05, inductor_value=inductance,
                                 inductor_tolerance=0.05, load_range=[9.5, 10.5], load_avg=load,
                                 sample_time=sample_time,
                                 switching_frequency=switch_period, source_voltage=source_voltage,
                                 reference_voltage=reference_voltage, desired_voltage=reference_voltage,
                                 max_action=np.array([1.]), min_action=np.array([0.]),
                                 max_state=np.array([1000., 1000.]), min_state=np.array([-100., -100.]), scale=1,
                                 max_init_state=np.array([3., 48.]), min_init_state=np.array([0., 0.]),
                                 evaluation_init=np.array([0., 0.]))

    # Create the learner
    learner = DDPG(runner=runner, num_training_steps=100000, time_per_step=sample_time, rollout_length=rollout_length,
                   evaluation_length=200,  # -1,
                   evaluation_iter=1,
                   num_evaluations=1, random_seed=1738,
                   replay_capacity=1000000, batch_size=64,
                   architecture='standard', actor_learning_rate=1e-4,
                   critic_learning_rate=1e-3, weight_decay=1e-2, gamma=0.99, tau=0.001, noise_sigma=0.2,
                   noise_theta=0.15,
                   log_path=path,
                   save_path=path + '/models',
                   load_path=None,
                   render_eval=False)

    learner.train_model()

    """ Save the last model as a .mat file """
    weights = [learner.actor.state_dict()['linear1.weight'].numpy(),
               learner.actor.state_dict()['linear2.weight'].numpy(),
               learner.actor.state_dict()['out.weight'].numpy()]
    biases = [learner.actor.state_dict()['linear1.bias'].numpy(),
              learner.actor.state_dict()['linear2.bias'].numpy(),
              learner.actor.state_dict()['out.bias'].numpy()]

    savemat(path + '/final_model.mat', mdict={'W': weights, 'b': biases})
