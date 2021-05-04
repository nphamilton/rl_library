"""
TODO header
"""
import numpy as np
from algorithms.ars.ars import *
from runners.f1_10 import *
from runners.f1_10_disc import *

if __name__ == '__main__':
    # Declare all the variables
    reference_file_name = 'track_porto_circuit.csv'
    lidar_sub_name = 'racecar/scan'
    odom_sub_name = 'racecar/odom'
    control_pub_name = 'racecar/drive_parameters'
    min_speed = 1.0
    max_speed = 1.1
    max_turning_angle = 34.0 * np.pi / 180.  # in radians
    training_steps = 10000000
    max_episode_len = 500
    log_path = './ars'
    # for the model training that worked, step size = 0.001, noise = 0.03, top perf = 4

    rospy.init_node('rl_control', anonymous=True)
    # Create the runner
    runner = F110Runner(reference_file_name=reference_file_name, lidar_sub_name=lidar_sub_name,
                        odom_sub_name=odom_sub_name, control_pub_name=control_pub_name, rate=10,
                        min_action=np.asarray([-max_turning_angle]),
                        max_action=np.asarray([max_turning_angle]), scale=0, lidar_angle=(np.pi / 2),
                        max_lidar_range=10.0, min_dist=0.1, crash_threshold=10)
    
    # runner = DiscreteF110Runner(reference_file_name=reference_file_name, lidar_sub_name=lidar_sub_name,
    #                             odom_sub_name=odom_sub_name, control_pub_name=control_pub_name, rate=10,
    #                             steering_scale=(np.pi / 6), num_categories=5,
    #                             lidar_angle=(np.pi / 2), max_lidar_range=10.0, min_dist=0.1)

    # Create the learner
    learner = ARS(runner=runner, num_training_steps=training_steps, step_size=0.001, dirs_per_iter=16, num_top_performers=4,
                 exploration_noise=0.03, rollout_length=max_episode_len, evaluation_length=max_episode_len, evaluation_iter=1,
                 num_evaluations=5, random_seed=1964, log_path=log_path, save_path=log_path+'/models', load_path=None, render_eval=True)

    learner.train_model()
