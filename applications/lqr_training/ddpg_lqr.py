"""
TODO header
"""
import numpy as np

from algorithms.ddpg.ddpg import *
from runners.lqr import *

if __name__ == '__main__':
    # Declare all the variables
    state_matrix = np.array([[1.01, 0.01, 0.0], [0.01, 1.01, 0.01], [0., 0.01, 1.01]])
    input_matrix = np.eye(3)
    state_cost = np.ones(3)
    input_cost = np.ones(3) / 1000
    lqr_horizon_length = 300

    # Create the runner
    runner = LQRRunner(state_matrix=state_matrix, input_matrix=input_matrix, state_cost=state_cost,
                       input_cost=input_cost, cross_term=None, horizon_length=lqr_horizon_length)

    # Create the learner
    learner = DDPG(runner=runner, num_training_steps=10000, time_per_step=1., rollout_length=lqr_horizon_length,
                   evaluation_length=lqr_horizon_length, evaluation_iter=1,
                   num_evaluations=1, random_seed=8, replay_capacity=10000, batch_size=64, actor_learning_rate=1e-4,
                   critic_learning_rate=1e-3, weight_decay=1e-2, gamma=0.99, tau=0.001, noise_sigma=0.2,
                   noise_theta=0.15,
                   log_path='/Users/nphamilton/rl_library/applications/lqr_training/ddpg',
                   save_path='/Users/nphamilton/rl_library/applications/lqr_training/ddpg/models', load_path=None)

    final_policy_reward_sum, execution_time, training_time = learner.train_model()
    print('Final Evaluation Reward: ' + str(final_policy_reward_sum))
    print('Time to run: ' + str(execution_time))
    print('Training time: ' + str(training_time))
    print('Time spent evaluating: ' + str(execution_time - training_time))
