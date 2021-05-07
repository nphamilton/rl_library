"""
TODO header
"""
import numpy as np
import pandas as pd
from scipy.io import savemat
from algorithms.ddpg.ddpg import *
from runners.avg_buck_converter import *
from runners.hybrid_buck_converter import *


def run_evaluation(runner, learner, eval_length, name):
    print(f'Evaluating {name}...')
    runner.reset(evaluate=True)
    obs = avg_runner.get_state()
    tot_reward = 0.0
    time = np.zeros(eval_length + 1)
    v_c = np.zeros_like(time)
    i_l = np.zeros_like(time)
    acc_reward = np.zeros_like(time)
    for i in range(eval_length):
        time[i] = i * sample_time
        i_l[i] = obs[3]
        v_c[i] = obs[2]
        acc_reward[i] = tot_reward
        obs, reward, _, _ = runner.step(learner.get_action(obs), render=False)
        tot_reward += reward
    time[eval_length] = eval_length * sample_time
    i_l[eval_length] = obs[3]
    v_c[eval_length] = obs[2]
    acc_reward[eval_length] = tot_reward

    data = pd.DataFrame({
        "time": time,
        "method": name,
        "current": i_l,
        "voltage": v_c,
        "accumulated reward": acc_reward,
    })
    print(f'{name} total reward: {tot_reward}')

    return data


def run_baseline_evaluation(runner, v_ref, v_s, eval_length):
    print('Evaluating baseline...')
    const_action = 2.0 * (v_ref / v_s) - 1.0  # must be mapped within the tanh range [-1, 1]
    runner.reset(evaluate=True)
    obs = avg_runner.get_state()
    tot_reward = 0.0
    time = np.zeros(eval_length + 1)
    v_c = np.zeros_like(time)
    i_l = np.zeros_like(time)
    acc_reward = np.zeros_like(time)
    for i in range(eval_length):
        time[i] = i * sample_time
        i_l[i] = obs[3]
        v_c[i] = obs[2]
        acc_reward[i] = tot_reward
        obs, reward, _, _ = runner.step(np.asarray([const_action]), render=False)
        tot_reward += reward
    time[eval_length] = eval_length * sample_time
    i_l[eval_length] = obs[3]
    v_c[eval_length] = obs[2]
    acc_reward[eval_length] = tot_reward

    data = pd.DataFrame({
        "time": time,
        "method": "baseline",
        "current": i_l,
        "voltage": v_c,
        "accumulated reward": acc_reward,
    })
    print(f'baseline total reward: {tot_reward}')

    return data

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
    eval_length = 200

    # Create the runners
    avg_runner = AvgBuckConverter(capacitor_value=capacitance, capacitor_tolerance=0.05, inductor_value=inductance,
                                  inductor_tolerance=0.05, load_range=[9.5, 10.5], load_avg=load,
                                  sample_time=sample_time,
                                  switching_frequency=switch_period, source_voltage=source_voltage,
                                  reference_voltage=reference_voltage, desired_voltage=reference_voltage,
                                  max_action=np.array([1.]), min_action=np.array([0.]),
                                  max_state=np.array([1000., 1000.]), min_state=np.array([-100., -100.]), scale=1,
                                  max_init_state=np.array([3., 48.]), min_init_state=np.array([0., 0.]),
                                  evaluation_init=np.array([0., 0.]))

    hybrid_runner = HybridBuckConverter(capacitor_value=capacitance, capacitor_tolerance=0.05,
                                        inductor_value=inductance,
                                        inductor_tolerance=0.05, load_range=[9.5, 10.5], load_avg=load,
                                        sample_time=sample_time,
                                        switching_frequency=switch_period, source_voltage=source_voltage,
                                        reference_voltage=reference_voltage, desired_voltage=reference_voltage,
                                        max_action=np.array([1.]), min_action=np.array([0.]),
                                        max_state=np.array([1000., 1000.]), min_state=np.array([-100., -100.]), scale=1,
                                        max_init_state=np.array([3., 48.]), min_init_state=np.array([0., 0.]),
                                        evaluation_init=np.array([0., 0.]))

    # Create the learners
    path_best_avg = './data/avg_dynamics/models/step_56000_model.pth'
    path_best_hybrid = './data/hybrid_dynamics/models/step_60000_model.pth'
    avg_learner_tanh = DDPG(runner=avg_runner, num_training_steps=100000, time_per_step=sample_time,
                            architecture='standard',
                            log_path='./ignore/',
                            save_path='./ignore/',
                            load_path=path_best_avg,
                            render_eval=True)
    avg_learner_hardtanh = DDPG(runner=avg_runner, num_training_steps=100000, time_per_step=sample_time,
                                architecture='verivital',
                                log_path='./ignore/',
                                save_path='./ignore/',
                                load_path=path_best_avg,
                                render_eval=True)
    hybrid_learner_tanh = DDPG(runner=avg_runner, num_training_steps=100000, time_per_step=sample_time,
                               architecture='standard',
                               log_path='./ignore/',
                               save_path='./ignore/',
                               load_path=path_best_hybrid,
                               render_eval=True)
    hybrid_learner_hardtanh = DDPG(runner=avg_runner, num_training_steps=100000, time_per_step=sample_time,
                                   architecture='verivital',
                                   log_path='./ignore/',
                                   save_path='./ignore/',
                                   load_path=path_best_hybrid,
                                   render_eval=True)

    # Make sure the best neural network controllers have been saved as .mat files for analysis in NNV
    weights = [avg_learner_tanh.actor.state_dict()['linear1.weight'].numpy(),
               avg_learner_tanh.actor.state_dict()['linear2.weight'].numpy(),
               avg_learner_tanh.actor.state_dict()['out.weight'].numpy()]
    biases = [avg_learner_tanh.actor.state_dict()['linear1.bias'].numpy(),
              avg_learner_tanh.actor.state_dict()['linear2.bias'].numpy(),
              avg_learner_tanh.actor.state_dict()['out.bias'].numpy()]

    savemat('./best_avg_nnc.mat', mdict={'W': weights, 'b': biases})

    weights = [hybrid_learner_tanh.actor.state_dict()['linear1.weight'].numpy(),
               hybrid_learner_tanh.actor.state_dict()['linear2.weight'].numpy(),
               hybrid_learner_tanh.actor.state_dict()['out.weight'].numpy()]
    biases = [hybrid_learner_tanh.actor.state_dict()['linear1.bias'].numpy(),
              hybrid_learner_tanh.actor.state_dict()['linear2.bias'].numpy(),
              hybrid_learner_tanh.actor.state_dict()['out.bias'].numpy()]

    savemat('./best_hybrid_nnc.mat', mdict={'W': weights, 'b': biases})

    # Collect all models performance in the averaged dynamics environment
    avg_baseline_data = run_baseline_evaluation(runner=avg_runner, v_ref=reference_voltage, v_s=source_voltage,
                                                eval_length=eval_length)
    avg_tanh_data = run_evaluation(runner=avg_runner, learner=avg_learner_tanh, eval_length=eval_length,
                                   name="avg_tanh")
    avg_hardtanh_data = run_evaluation(runner=avg_runner, learner=avg_learner_hardtanh, eval_length=eval_length,
                                       name="avg_hardtanh")
    hybrid_tanh_data = run_evaluation(runner=avg_runner, learner=hybrid_learner_tanh, eval_length=eval_length,
                                      name="hybrid_tanh")
    hybrid_hardtanh_data = run_evaluation(runner=avg_runner, learner=hybrid_learner_hardtanh, eval_length=eval_length,
                                          name="hybrid_hardtanh")
    avg_dynamics_data = pd.concat([avg_baseline_data, avg_tanh_data, avg_hardtanh_data, hybrid_tanh_data,
                                   hybrid_hardtanh_data])
    avg_dynamics_data.to_csv('./data/avg_dynamics_evaluations.csv')

    # Collect all models performance in the hybrid dynamics environment
    avg_baseline_data = run_baseline_evaluation(runner=hybrid_runner, v_ref=reference_voltage, v_s=source_voltage,
                                                eval_length=eval_length)
    avg_tanh_data = run_evaluation(runner=hybrid_runner, learner=avg_learner_tanh, eval_length=eval_length,
                                   name="avg_tanh")
    avg_hardtanh_data = run_evaluation(runner=hybrid_runner, learner=avg_learner_hardtanh, eval_length=eval_length,
                                       name="avg_hardtanh")
    hybrid_tanh_data = run_evaluation(runner=hybrid_runner, learner=hybrid_learner_tanh, eval_length=eval_length,
                                      name="hybrid_tanh")
    hybrid_hardtanh_data = run_evaluation(runner=hybrid_runner, learner=hybrid_learner_hardtanh, eval_length=eval_length,
                                          name="hybrid_hardtanh")
    hybrid_dynamics_data = pd.concat([avg_baseline_data, avg_tanh_data, avg_hardtanh_data, hybrid_tanh_data,
                                   hybrid_hardtanh_data])
    hybrid_dynamics_data.to_csv('./data/hybrid_dynamics_evaluations.csv')
