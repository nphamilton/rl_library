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
    state_cost = np.ones(3) / 1000
    input_cost = np.ones(3)
    lqr_horizon_length = 300

    # Create the runner
    runner = LQRRunner(state_matrix=state_matrix, input_matrix=input_matrix, state_cost=state_cost,
                       input_cost=input_cost, cross_term=None, horizon_length=lqr_horizon_length)

    # Create the learner
    learner = ARS(runner=runner, num_training_steps=1000000000, step_size=0.02, dirs_per_iter=16, num_top_performers=16,
                  exploration_noise=0.03, rollout_length=1000, evaluation_length=1000, evaluation_iter=10,
                  num_evaluations=5, random_seed=8, log_path='.', save_path='.')

    final_policy_reward_sum, execution_time, training_time = learner.train_model()
    print('Final Evaluation Reward: ' + str(final_policy_reward_sum))
    print('Time to run: ' + str(execution_time))
    print('Training time: ' + str(training_time))
    print('Time spent evaluating: ' + str(execution_time - training_time))
