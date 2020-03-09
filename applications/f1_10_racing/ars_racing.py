"""
TODO header
"""
import numpy as np
from algorithms.ars.ars import *
from runners.f1_10 import *

if __name__ == '__main__':
    # Declare all the variables
    reference_file_name = ''
    lidar_sub_name = ''
    odom_sub_name = ''
    control_pub_name = ''
    min_speed = 1.0
    max_speed = 3.0
    max_turning_angle = 34.0 * np.pi / 180.  # in radians

    # Create the runner
    runner = F110Runner(reference_file_name=reference_file_name, lidar_sub_name=lidar_sub_name,
                        odom_sub_name=odom_sub_name, control_pub_name=control_pub_name, rate=10,
                        min_action=np.asarray([min_speed, -max_turning_angle]),
                        max_action=np.asarray([max_speed, max_turning_angle]), scale=0, lidar_angle=(np.pi / 2),
                        max_lidar_range=10.0, min_dist=0.1, crash_threshold=10)

    # Create the learner
    learner = ARS(runner=runner, num_training_steps=100000, step_size=0.02, dirs_per_iter=16, num_top_performers=16,
                  exploration_noise=0.03, rollout_length=1000, evaluation_length=1000, evaluation_iter=10,
                  num_evaluations=5, random_seed=8, log_path='.', save_path='.')

    learner.train_model()
