"""
TODO header
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from algorithms.ddpg.ddpg import *
from runners.avg_buck_converter import *
from runners.hybrid_buck_converter import *

if __name__ == '__main__':
    # Declare all the variables
    sample_time = 0.001
    rollout_length = 2000

    # Create the runner
    path = '/Users/nphamilton/rl_library/applications/buck_converter/ddpg'
    runner = AvgBuckConverter(capacitor_value=2.2e-3, capacitor_tolerance=0.05, inductor_value=2.65e-3,
                              inductor_tolerance=0.05, load_range=[9.5, 10.5], load_avg=10.0,
                              sample_time=sample_time,
                              switching_frequency=1e4, source_voltage=100.0, reference_voltage=48.0,
                              desired_voltage=48.0,
                              max_action=np.array([1.]), min_action=np.array([0.]),
                              max_state=np.array([1000., 1000.]), min_state=np.array([-100., -100.]), scale=1,
                              max_init_state=np.array([3., 48.]), min_init_state=np.array([0., 0.]),
                              evaluation_init=np.array([0., 0.]))

    # path = '/Users/nphamilton/rl_library/applications/buck_converter/ddpg_hybrid'
    # runner = HybridBuckConverter(capacitor_value=2.2e-3, capacitor_tolerance=0.05, inductor_value=2.65e-3,
    #                              inductor_tolerance=0.05, load_range=[9.5, 10.5], load_avg=10.0,
    #                              sample_time=sample_time,
    #                              switching_frequency=1e4, source_voltage=100.0, reference_voltage=48.0,
    #                              desired_voltage=48.0,
    #                              max_action=np.array([1.]), min_action=np.array([0.]),
    #                              max_power=3000., scale=1,
    #                              max_init_state=np.array([3., 48.]), min_init_state=np.array([0., 0.]),
    #                              evaluation_init=np.array([0., 0.]))

    # runner.reset(evaluate=True)
    # for i in range(200):
    #     runner.step(np.asarray([1.]), render=True)
    #
    # """ Plot an evaluation """
    # step = 0
    # reward_sum = 0
    #
    # # Start the evaluation from a safe starting point
    # runner.reset(evaluate=True)
    # state = runner.get_state()
    # done = 0
    # exit_cond = 0
    # voltages = [state[2]]
    # currents = [state[3]]
    #
    # while runner.is_available():
    #     # Stop the controller if there is a collision or time-out
    #     if done or exit_cond:
    #         # stop
    #         runner.stop()
    #         break
    #
    #     # Determine the next action
    #     action = np.asarray([0.6])  # learner.get_action(state)  # No noise injected during evaluation
    #
    #     # Execute determined action
    #     next_state, reward, done, exit_cond = runner.step(action)
    #     currents.append(next_state[3])
    #     voltages.append(next_state[2])
    #
    #     # Update for next step
    #     reward_sum += reward
    #     state = next_state
    #     step += 1
    #
    # print('Total reward: ' + str(reward_sum))
    # fig, ax = plt.subplots()
    # ax.plot(currents, voltages)
    # ax.set(xlim=[0.0, 2.5], ylim=[0.0, 8.0])
    # plt.show()


    # path = '/Users/nphamilton/rl_library/applications/buck_converter/ddpg_hybrid'
    # runner = HybridBuckConverter(capacitor_value=2.2e-3, capacitor_tolerance=0.05, inductor_value=2.65e-3,
    #                              inductor_tolerance=0.05, load_range=[9.5, 10.5], load_avg=10.0,
    #                              sample_time=sample_time,
    #                              switching_frequency=1e4, source_voltage=100.0, reference_voltage=48.0,
    #                              desired_voltage=48.0,
    #                              max_action=np.array([1.]), min_action=np.array([0.]),
    #                              max_power=3000., scale=1,
    #                              max_init_state=np.array([3., 48.]), min_init_state=np.array([0., 0.]),
    #                              evaluation_init=np.array([0., 0.]))

    # runner.reset(evaluate=True)
    # for i in range(200):
    #     runner.step(np.asarray([1.]), render=True)
    #
    # """ Plot an evaluation """
    # step = 0
    # reward_sum = 0
    #
    # # Start the evaluation from a safe starting point
    # runner.reset(evaluate=True)
    # state = runner.get_state()
    # done = 0
    # exit_cond = 0
    # voltages = [state[2]]
    # currents = [state[3]]
    #
    # while runner.is_available():
    #     # Stop the controller if there is a collision or time-out
    #     if done or exit_cond:
    #         # stop
    #         runner.stop()
    #         break
    #
    #     # Determine the next action
    #     action = np.asarray([0.6])  # learner.get_action(state)  # No noise injected during evaluation
    #
    #     # Execute determined action
    #     next_state, reward, done, exit_cond = runner.step(action)
    #     currents.append(next_state[3])
    #     voltages.append(next_state[2])
    #
    #     # Update for next step
    #     reward_sum += reward
    #     state = next_state
    #     step += 1
    #
    # print('Total reward: ' + str(reward_sum))
    # fig, ax = plt.subplots()
    # ax.plot(currents, voltages)
    # ax.set(xlim=[0.0, 2.5], ylim=[0.0, 8.0])
    # plt.show()


    # Create the learner
    learner = DDPG(runner=runner, num_training_steps=100000, time_per_step=sample_time, rollout_length=rollout_length,
                   evaluation_length=200,  # -1,
                   evaluation_iter=1,
                   num_evaluations=1, random_seed=1738, replay_capacity=1000000, batch_size=64,
                   architecture='verivital', actor_learning_rate=1e-4,
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

    # """ Plot an evaluation """
    # step = 0
    # reward_sum = 0
    #
    # # Start the evaluation from a safe starting point
    # runner.reset(evaluate=True)
    # state = runner.get_state()
    # done = 0
    # exit_cond = 0
    # voltages = [state[2]]
    # currents = [state[3]]
    #
    # while runner.is_available():
    #     # Stop the controller if there is a collision or time-out
    #     if done or exit_cond:
    #         # stop
    #         runner.stop()
    #         break
    #
    #     # Determine the next action
    #     action = np.asarray([0.6])  # learner.get_action(state)  # No noise injected during evaluation
    #
    #     # Execute determined action
    #     next_state, reward, done, exit_cond = runner.step(action)
    #     currents.append(next_state[3])
    #     voltages.append(next_state[2])
    #
    #     # Update for next step
    #     reward_sum += reward
    #     state = next_state
    #     step += 1
    #
    # print('Total reward: ' + str(reward_sum))
    # fig, ax = plt.subplots()
    # ax.plot(currents, voltages)
    # ax.set(xlim=[0.0, 2.5], ylim=[0.0, 8.0])
    # plt.show()
