"""
TODO header
"""
import numpy as np

from algorithms.ars.ars import *
from runners.gym_runner import *

if __name__ == '__main__':
    # Declare all the variables
    state_matrix = np.array([[1.01, 0.01, 0.0], [0.01, 1.01, 0.01], [0., 0.01, 1.01]])
    input_matrix = np.eye(3)
    state_cost = np.ones(3) / 1000
    input_cost = np.ones(3)
    lqr_horizon_length = 300
    environment_name = 'CartPole-v1'  # 'MountainCarContinuous-v0'  # 'Pendulum-v0'

    # Create the runner
    runner = GymRunner(env_name=environment_name, render=False)

    # Create the learner
    learner = ARS(runner=runner, num_training_steps=1000000, step_size=0.02, dirs_per_iter=24, num_top_performers=8,
                  exploration_noise=0.02, rollout_length=lqr_horizon_length, evaluation_length=500,
                  evaluation_iter=1,
                  num_evaluations=1, random_seed=8,
                  log_path='/Users/nphamilton/rl_library/applications/gym_training/ars_log',
                  save_path='/Users/nphamilton/rl_library/applications/gym_training/ars_models')

    final_policy_reward_sum, execution_time, training_time = learner.train_model()
    print('Final Evaluation Reward: ' + str(final_policy_reward_sum))
    print('Time to run: ' + str(execution_time))
    print('Training time: ' + str(training_time))
    print('Time spent evaluating: ' + str(execution_time - training_time))
