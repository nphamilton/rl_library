"""
TODO header
"""

import numpy as np
from algorithms.ars.ars import *
from runners.gym_runner import *
from runners.gym_pendulum import *

if __name__ == '__main__':
    # Declare all the variables
    rollout_length = 1000
    environment_name = 'Pendulum-v0'
    path = '/Users/nphamilton/rl_library/applications/mlv_project/inverted_pendulum/ars'

    # Create the runner
    # runner = GymRunner(env_name=environment_name, scale=1, render=True)
    runner = GymPendulum(scale=0, render=True)

    # Create the learner
    learner = ARS(runner=runner, num_training_steps=1000000, step_size=0.02, dirs_per_iter=16, num_top_performers=16,
                  exploration_noise=0.02, rollout_length=rollout_length, evaluation_length=-1,
                  evaluation_iter=1,
                  num_evaluations=1, random_seed=1964,
                  log_path=path,
                  save_path=path + '/models')

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
