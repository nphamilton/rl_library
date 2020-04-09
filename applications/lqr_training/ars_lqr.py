"""
TODO header
"""
import numpy as np

from algorithms.ars.ars import *
from runners.lqr import *

if __name__ == '__main__':
    # Declare all the variables
    state_matrix = np.array([[1.01, 0.01, 0.0], [0.01, 1.01, 0.01], [0., 0.01, 1.01]])
    input_matrix = np.eye(3)
    state_cost = np.ones(3) / 100
    input_cost = np.ones(3) / 1000
    max_action = np.array([1])
    max_state = np.array([1000., 1000., 1000.])
    lqr_horizon_length = 300
    eval_init = np.array([1.33828699, -2.61368526, -1.85276285])

    # Create the runner
    runner = LQRRunner(state_matrix=state_matrix, input_matrix=input_matrix, state_cost=state_cost,
                       input_cost=input_cost, cross_term=None, max_action=max_action, min_action=None,
                       max_state=max_state, min_state=None, scale=1, horizon_length=lqr_horizon_length,
                       evaluation_init=eval_init)

    # Create the learner
    learner = ARS(runner=runner, num_training_steps=1000000, step_size=0.02, dirs_per_iter=16, num_top_performers=16,
                  exploration_noise=0.02, rollout_length=lqr_horizon_length, evaluation_length=lqr_horizon_length,
                  evaluation_iter=1,
                  num_evaluations=1, random_seed=1946,
                  log_path='/Users/nphamilton/rl_library/applications/lqr_training/ars',
                  save_path='/Users/nphamilton/rl_library/applications/lqr_training/ars/models')

    final_policy_reward_sum, execution_time, training_time = learner.train_model()
    print('Final Evaluation Reward: ' + str(final_policy_reward_sum))
    print('Time to run: ' + str(execution_time))
    print('Training time: ' + str(training_time))
    print('Time spent evaluating: ' + str(execution_time - training_time))
