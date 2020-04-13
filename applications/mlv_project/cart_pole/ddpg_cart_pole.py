"""
TODO header
"""

import numpy as np
from algorithms.ddpg.ddpg import *
from runners.lqr import *

if __name__ == '__main__':
    # Declare all the variables
    state_matrix = np.matrix([[0., 1., 0., 0.], [0., 0., 0.716, 0.], [0., 0., 0., 1.], [0., 0., 15.76, 0.]])
    input_matrix = np.matrix([[0.], [0.9755], [0.], [1.46]])
    state_cost = np.ones(4)
    input_cost = np.ones(1) * 0.0005
    max_init = np.array([0.05, 0.1, 0.05, 0.05])
    max_state = np.array([.3, .5, .3, .5])
    max_action = np.array([15.])
    lqr_horizon_length = 200
    rollout_length = 1000
    path = '/Users/nphamilton/rl_library/applications/mlv_project/cart_pole/ddpg'

    # Create the runner
    runner = LQRRunner(state_matrix=state_matrix, input_matrix=input_matrix, state_cost=state_cost,
                       input_cost=input_cost, cross_term=None, scale=0, horizon_length=lqr_horizon_length,
                       max_init_state=max_init, max_action=max_action,
                       max_state=max_state, evaluation_init=None)

    # Create the learner
    learner = DDPG(runner=runner, num_training_steps=100000, time_per_step=1., rollout_length=rollout_length,
                   evaluation_length=-1, evaluation_iter=1,
                   num_evaluations=1, random_seed=1964, replay_capacity=10000, batch_size=64, actor_learning_rate=1e-4,
                   critic_learning_rate=1e-3, weight_decay=1e-2, gamma=0.99, tau=0.001, noise_sigma=0.2,
                   noise_theta=0.15,
                   log_path=path,
                   save_path=path + '/models',
                   load_path=None)

    final_policy_reward_sum, execution_time, training_time = learner.train_model()
    print('Final Evaluation Reward: ' + str(final_policy_reward_sum))
    print('Time to run: ' + str(execution_time))
    print('Training time: ' + str(training_time))
    print('Time spent evaluating: ' + str(execution_time - training_time))

    # Collect final evaluation information
    num_evals = 1000
    rewards = np.zeros(num_evals)
    steps = np.zeros(num_evals)
    dones = np.zeros(num_evals)
    exits = np.zeros(num_evals)
    runner.render = False
    for i in range(num_evals):
        rewards[i], steps[i], dones[i], exits[i] = learner.evaluate_model(-1)

    # Save the results
    f = open(path + '/final_eval.csv', "w+")
    f.write("Avg Reward, Reward std, Avg Steps, Steps std, Percent Dones, Percent Exits \n")
    f.write(f'{np.mean(rewards)}, {np.std(rewards)}, {np.mean(steps)}, {np.std(steps)}, {np.sum(dones) / num_evals}, '
            f'{np.sum(exits) / num_evals}')
    f.close()
