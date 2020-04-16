"""
TODO header
"""

import numpy as np
from algorithms.ddpg.ddpg import *
from runners.gym_runner import *
from runners.gym_pendulum import *

if __name__ == '__main__':
    # Declare all the variables
    rollout_length = 1000
    environment_name = 'Pendulum-v0'
    path = '/Users/nphamilton/rl_library/applications/mlv_project/inverted_pendulum/ddpg'

    # Create the runner
    # runner = GymRunner(env_name=environment_name, scale=1, render=True)
    runner = GymPendulum(scale=1, render=False)

    # Create the learner
    learner = DDPG(runner=runner, num_training_steps=100000, time_per_step=1., rollout_length=rollout_length,
                   evaluation_length=-1, evaluation_iter=1,
                   num_evaluations=5, random_seed=1964, replay_capacity=10000, batch_size=64, actor_learning_rate=1e-4,
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
