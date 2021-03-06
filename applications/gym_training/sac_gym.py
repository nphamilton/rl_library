"""
TODO header
"""

from algorithms.sac.sac import *
from runners.gym_runner import *

if __name__ == '__main__':
    # Declare all the variables
    rollout_length = 1000
    environment_name = 'Pendulum-v0'  # 'MountainCarContinuous-v0'  # 'MountainCarContinuous-v0'  # 'Pendulum-v0'

    # Create the runner
    runner = GymRunner(env_name=environment_name, scale=1, render=True)

    # Create the learner
    learner = SAC(runner=runner, num_training_steps=100000, rollout_length=rollout_length,
                  evaluation_length=-1, evaluation_iter=1,
                  num_evaluations=1, random_seed=1964, replay_capacity=10000, batch_size=64, learning_rate=1e-4,
                  alpha=1.0, gamma=0.99, tau=0.005,
                  log_path='/Users/nphamilton/rl_library/applications/gym_training/sac',
                  save_path='/Users/nphamilton/rl_library/applications/gym_training/sac/models',
                  load_path=None)

    final_policy_reward_sum, execution_time, training_time = learner.train_model()
    print('Final Evaluation Reward: ' + str(final_policy_reward_sum))
    print('Time to run: ' + str(execution_time))
    print('Training time: ' + str(training_time))
    print('Time spent evaluating: ' + str(execution_time - training_time))
