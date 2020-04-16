"""
TODO header
"""

from algorithms.ppo.ppo import *
from runners.gym_runner import *

if __name__ == '__main__':
    # Declare all the variables
    rollout_length = 2048
    environment_name = 'Pendulum-v0'  #'MountainCarContinuous-v0'  # 'CartPole-v1'  # 'Pendulum-v0'

    # Create the runner
    runner = GymRunner(env_name=environment_name, scale=1, render=True)

    # Create the learner
    learner = PPO(runner=runner, num_training_steps=100000, rollout_length=rollout_length,
                  evaluation_length=-1, evaluation_iter=1,
                  num_evaluations=5, random_seed=1964, minibatch_size=64, num_epochs=10,
                  learning_rate=1e-4, discount_gamma=0.995, gae_lambda=0.97, clip_param=0.2,
                  log_path='/Users/nphamilton/rl_library/applications/gym_training/ppo',
                  save_path='/Users/nphamilton/rl_library/applications/gym_training/ppo/models',
                  load_path=None, render_eval=False)

    final_policy_reward_sum, execution_time, training_time = learner.train_model()
    print('Final Evaluation Reward: ' + str(final_policy_reward_sum))
    print('Time to run: ' + str(execution_time))
    print('Training time: ' + str(training_time))
    print('Time spent evaluating: ' + str(execution_time - training_time))
