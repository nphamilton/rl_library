"""
TODO header
"""
import numpy as np
from algorithms.ddpg.ddpg import *
from runners.f1_10 import *

if __name__ == '__main__':
    # Declare all the variables
    reference_file_name = 'track_porto_circuit.csv'
    lidar_sub_name = 'racecar/scan'
    odom_sub_name = 'racecar/odom'
    control_pub_name = 'racecar/drive_parameters'
    min_speed = 1.0
    max_speed = 1.1
    max_turning_angle = 34.0 * np.pi / 180.  # in radians
    training_steps = 45000
    max_episode_len = 500
    log_path = './ddpg'

    rospy.init_node('rl_control', anonymous=True)
    # Create the runner
    runner = F110Runner(reference_file_name=reference_file_name, lidar_sub_name=lidar_sub_name,
                        odom_sub_name=odom_sub_name, control_pub_name=control_pub_name, rate=10,
                        min_action=np.asarray([-max_turning_angle]),
                        max_action=np.asarray([max_turning_angle]), scale=1, lidar_mode=2,
                        lidar_angle=(np.pi / 2),
                        max_lidar_range=10.0, min_dist=0.2, crash_threshold=10)

    # Create the learner
    learner = DDPG(runner=runner, num_training_steps=training_steps, time_per_step=0.1,
                  rollout_length=max_episode_len,
                  evaluation_length=500, evaluation_iter=1,
                  num_evaluations=1, random_seed=15,
                  replay_capacity=1000000, batch_size=64,
                  actor_learning_rate=1e-4, critic_learning_rate=1e-3, weight_decay=1e-2, gamma=.99, tau=.001, 
                  noise_sigma=0.2, noise_theta=0.15, log_path=log_path,
                  save_path=log_path + '/models',
                  load_path=None)

    learner.train_model()
