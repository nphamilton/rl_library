"""
File:   f1_10.py
Author: Nathaniel Hamilton

Description: This class implements a runner for the F1/10th racing simulator. The goal of an agent using this runner is
             to complete laps as quickly as possible.

Usage:       Import the entire class file to instantiate and use this runner.

"""
import numpy as np
import pandas as pd
import rospy
import copy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from race.msg import drive_param

# from runners.abstract_runner import Runner


class F110Runner():
    def __init__(self, reference_file_name, lidar_sub_name, odom_sub_name, control_pub_name, rate=10,
                 min_action=np.asarray([(-34.0 * np.pi / 180.)]),
                 max_action=np.asarray([(34.0 * np.pi / 180.)]), scale=0, lidar_mode=0,
                 lidar_angle=(np.pi / 2), max_lidar_range=10.0, min_dist=0.1, crash_threshold=10):
        """
        TODO: describe the runner in detail
        :param reference_file_name:
        :param lidar_sub_name:
        :param odom_sub_name:
        :param control_pub_name:
        :param rate:
        :param min_action:
        :param max_action:
        :param scale:
        :param lidar_mode:          (int)   A value that determines which mode the lidar is being used. 
                                            If 0, the full range of readings within +-lidar_angle are 
                                            used. If 1, 5 equidistant readings between +-90deg are used.
                                            If 2, 7 readings are used. The 5 from mode 1 are the same, 
                                            with 2 additional from +-30deg.
        :param lidar_angle:
        :param max_lidar_range:
        :param min_dist:
        :param crash_threshold:
        """

        # Save the relevant parameters
        self.rate = rospy.Rate(rate)
        self.min_action = min_action
        self.max_action = max_action
        self.scale = scale
        self.lidar_mode = lidar_mode
        self.lidar_angle = lidar_angle
        self.max_lidar_range = max_lidar_range
        self.min_dist = min_dist
        self.crash_threshold = crash_threshold

        self.scale_mult = (max_action - min_action) / 2.0
        self.scale_add = (max_action - min_action) / 2.0 + min_action

        self.obs_shape = None
        self.action_shape = min_action.shape
        self.is_discrete = False

        if lidar_mode == 2:
            self.target_angles = np.asarray([-90., -60., -45., -30, 0., 30., 45., 60., 90.]) * (np.pi / 180.)
        else:
           self.target_angles = np.asarray([-90., -45., 0., 45., 90.]) * (np.pi / 180.)
        self.indices = None
        self.prev_reward = 0.0

        # Initialize subscribers
        rospy.Subscriber(lidar_sub_name, LaserScan, self.__callback_lidar)
        rospy.Subscriber(odom_sub_name, Odometry, self.__callback_odom)

        # Initialize publishers
        self.pub_drive_param = rospy.Publisher(control_pub_name, drive_param, queue_size=5)

        # Setup function for resetting the environment
        self.reset_env = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Load the reference trajectory used for calculating the reward.
        self.__load_reference_track(reference_file_name)
        self.prev_pos_index = 0

        self.rate.sleep()
        self.rate.sleep()
        self.rate.sleep()

    def __calculate_reward(self):
        """
        TODO: write up explanation
        :return:
        """

        # Determine the current position index
        curr_pos_index, dist_from_point = self.__find_closest_point(self.prev_pos_index)

        # Calculate the distance traveled between points
        reward = 1*(self.ref_track[curr_pos_index, 2] - self.ref_track[self.prev_pos_index, 2])
        # reward = curr_pos_index - self.prev_pos_index
        # print(curr_pos_index)

        # The reward will be a very large negative value if a lap was completed, so recompute in those cases
        if reward < -15.0:
            end = len(self.ref_track) - 1
            dist_to_end = self.ref_track[end, 2] - self.ref_track[self.prev_pos_index, 2]
            dist_from_start = self.ref_track[curr_pos_index, 2]
            reward = dist_to_end + dist_from_start
            # print('Special case reward: ' + str(reward))
        # If an unusually large reward occurs, then the agent drove backwards across the finish line
        if reward > 15.0:
            end = len(self.ref_track) - 1
            dist_to_end = self.ref_track[end, 2] - self.ref_track[curr_pos_index, 2]
            dist_from_start = self.ref_track[self.prev_pos_index, 2]
            reward = -1 * (dist_to_end + dist_from_start)
            # print('Special case reward: ' + str(reward))
        
        # reward += -0.5 * dist_from_point

        self.prev_pos_index = copy.copy(curr_pos_index)

        if reward < 0.0:
            # print(reward)
            reward = -1.0

        return reward, dist_from_point

    def __callback_odom(self, data):
        """
        TODO: decide what info you want from odom
        """
        qx = data.pose.pose.orientation.x
        qy = data.pose.pose.orientation.y
        qz = data.pose.pose.orientation.z
        qw = data.pose.pose.orientation.w

        quaternion = (qx, qy, qz, qw)
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]

        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

        dx = data.twist.twist.linear.x
        dy = data.twist.twist.linear.y
        speed = np.sqrt(dx ** 2 + dy ** 2)

        # self.prev_pos = copy.copy(self.pos)
        self.pos = np.asarray([x, y, yaw, speed])
        # print('speed' + str(speed))

    def __callback_lidar(self, data):
        """
        This callback method responds when a lidar scan is received. The lidar data is clipped to eliminate extraneous
        maximums. The result is saved as the state.

        :param data:    (sensor_msgs.LaserScan) The lidar data message
        """
        # Collect the ranges
        ranges = np.asarray(data.ranges)

        # If the indices of ranges hasn't been determined yet, determine them
        if self.indices is None:
            min_angle = data.angle_min
            max_angle = data.angle_max
            angle_step = data.angle_increment

            # Create an array of the measured angles
            angles = np.arange(min_angle, max_angle, angle_step, dtype=np.float32)

            # Compute the indices of the desired lidar angles
            if self.lidar_mode == 0:
                # Compute the indices of values within +/- the desired lidar angle
                self.indices = np.where(((-1 * self.lidar_angle) <= angles) & (angles <= self.lidar_angle))
                self.obs_shape = (len(self.indices[0]), 1)
            else:
                self.indices = range(len(self.target_angles))
                for i in range(len(self.target_angles)):
                    self.indices[i] = int(round((self.target_angles[i] - min_angle) / angle_step))
                print(angles[self.indices])
                self.obs_shape = (len(self.indices), 1)
            

        # Clip any range values larger than the set maximum since the lidar data is very noisy at larger distances
        max_ranges = self.max_lidar_range * np.ones_like(ranges)
        clipped_ranges = np.minimum(ranges, max_ranges)

        self.lidar_ranges = clipped_ranges

        return

    def __find_closest_point(self, prev_index):
        """
        TODO: write explanation
        :param prev_index:
        :return:
        """

        # Record the current position
        curr_pos = self.pos
        ref = self.ref_track
        l = len(ref)

        # Find the closest point using a binary search-esque method
        index_min = (prev_index - 10) % l
        index_max = (prev_index + 20) % l
        select_index = index_min
        min_dist = 10000.0
        iters = 0
        while iters < 5:
            if index_max > index_min:
                for i in range(index_min, index_max):
                    dist = np.sqrt((ref[i, 0] - curr_pos[0])**2 + (ref[i, 1] - curr_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        select_index = i
            else:
                for i in range(index_max):
                    dist = np.sqrt((ref[i, 0] - curr_pos[0]) ** 2 + (ref[i, 1] - curr_pos[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        select_index = i
                for i in range(index_min, l):
                    dist = np.sqrt((ref[i, 0] - curr_pos[0]) ** 2 + (ref[i, 1] - curr_pos[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        select_index = i
            # TODO: explain why
            if abs(select_index - index_min) < 3:
                index_max = (index_min + 5) % l
                index_min = (index_min - 30) % l
            elif abs(select_index - index_max) < 3:
                index_min = (index_max - 5) % l
                index_max = (index_max + 30) % l
            else:
                break
            iters += 1

        return select_index, min_dist

    def get_state(self):
        """
        This function returns the current state of the agent in the environment.

        :return curr_state: (np.array) The current state of the agent is the selected lidar values.
        """
        curr_state = self.lidar_ranges[self.indices]
        # curr_state = (self.lidar_ranges[self.indices] - 5.) / 5.
        # print(curr_state)

        return curr_state

    def is_available(self):
        """
        This method checks to make sure the environment is still available. Some environments are able to discontinue
        without stopping the learning process.

        :input:
            None
        :output:
            return T/F
        """

        return not rospy.is_shutdown()

    def __is_terminal(self, reward):
        """
        This method determines whether or not the agent is currently in a terminal state, i.e. done, or exit_cond.

        In this environment, the agent can never reach a "done" state, so done is always 0.

        The agent is in an exit condition, exit_cond, if the current lidar reading has multiple reading below a set
        threshold. This would suggest the car has crashed.

        :return done:       (int) 1 if a done condition has been reached, 0 otherwise
        :return exit_cond:  (int) 1 if a fatal condition has been reached, 0 otherwise
        """

        # Initialize the terminal signals to false
        done = 0
        exit_cond = 0

        # Find readings that are below the set minimum. If there are multiple readings below the threshold, a crash
        # likely occurred and the episode should end
        # print(self.lidar_ranges)
        # indices = np.where(self.lidar_ranges <= self.min_dist)
        # # print(indices)
        # if len(indices[0]) >= self.crash_threshold:
        #     exit_cond = 1
        # If the speed is less than 0.3, then the vehicle is pressed against a wall and not moving. Thus, it has crashed.
        # print('Speed: ' + str(self.pos[3]))
        if self.pos[3] < 0.2:
            dist_to_start = np.sqrt((self.pos[0] -self.ref_track[1088, 0])**2 + (self.pos[1] -self.ref_track[1088, 1])**2)
            # print(dist_to_start)
            if dist_to_start > 0.4:
                exit_cond = 1

        if reward <= -1.0:
            exit_cond = 1

        return done, exit_cond

    def __load_reference_track(self, file_name):
        """
        TODO: documentation
        :param file_name:
        :return:
        """

        # Read the waypoint file
        df = pd.read_csv(file_name, names=['X', 'Y', 'Z', 'W', 'P'])
        waypoints = df[['X', 'Y']].to_numpy()

        # Compute the distances along the track
        ref_track = np.zeros((len(waypoints), 3), dtype=float)
        ref_track[0, :] = [waypoints[0, 0], waypoints[0, 1], 0.0]
        tot_dist = 0.0
        for i in range(1, len(waypoints)):
            dist = np.sqrt(
                (waypoints[i, 0] - waypoints[(i - 1), 0]) ** 2 + (waypoints[i, 1] - waypoints[(i - 1), 1]) ** 2)
            tot_dist += dist
            ref_track[i, :] = [waypoints[i, 0], waypoints[i, 1], tot_dist]

        # print('Track length: ' + str(tot_dist))
        self.ref_track = ref_track

        return

    def __publish_cmd(self, velocity, steering_angle):
        """
        TODO: documentation
        :param velocity:
        :param steering_angle:
        :return:
        """

        msg = drive_param()
        msg.angle = steering_angle
        msg.velocity = velocity
        self.pub_drive_param.publish(msg)

        return

    def step(self, action, render=False):
        """
        This function should execute a single step within the environment and return all necessary information
        including in the following order:
            1 next state/observation
            2 reward
            3 done (if the agent has reached a terminal state, this will be 1, otherwise 0)
            4 exit condition (if the agent has reached a fatal state, this will be 1, otherwise 0)

        :input:
            action
        :output:
            return next_state, reward, done, exit
        """
        if self.scale == 1:
            # Scale the action
            action = np.multiply(action, self.scale_mult) + self.scale_add
        elif self.scale == 0:
            action = np.minimum(np.maximum(action, self.min_action), self.max_action)
        else:
            raise NotImplementedError

        # Publish action
        vel_cmd = 1.0  # action[0]
        steer_cmd = action  # [1]
        self.__publish_cmd(vel_cmd, steer_cmd)

        # Wait specified time
        self.rate.sleep()

        # Collect new state
        next_state = self.get_state()
        reward, min_dist = self.__calculate_reward()
        # print(reward)
        done, exit_cond = self.__is_terminal(reward)

        # if not render:
        #     # Reward for following the waypoints closely
        #     reward = max((1. - min_dist) / 10., 0.0) + self.prev_reward
        #     self.prev_reward = reward
        #     if exit_cond:
        #         reward = 0.0  # -10.0

        # else:
        #     if exit_cond:
        #         reward = -1.0
        
        if exit_cond:
                reward = 0.0

        # reward *= 10
        # print(reward)
        return next_state, reward, done, exit_cond

    def stop(self):
        """
        This method stops the car.
        """
        self.__publish_cmd(0.0, 0.0)

        return

    def reset(self, evaluate=False):
        """
        This function resets the simulation environment.
        """
        self.reset_env()
        for _ in range(20):
            self.__publish_cmd(0.0, 0.0)
            self.rate.sleep()
        self.__publish_cmd(1.0, 0.0)
        self.rate.sleep()
        self.prev_pos_index, _ = self.__find_closest_point(1088)
        self.prev_reward = 0.0
        
        
        # If this run isn't being evaluated, we need to add some randomness to the starting point or else the agents will not learn well
        # if not evaluate:
        #     random_steps = np.random.randint(0, 20)
        #     ref = self.ref_track
        #     l = len(ref)
        #     curr_pos_index, _ = self.__find_closest_point(self.prev_pos_index)
        #     for _ in range(random_steps):
        #         ranges = self.get_state()
        #         max_range = np.max(ranges)
        #         indices = np.where(ranges>=max_range)
        #         target_index = np.mean(indices)
        #         angle = ((2 * self.lidar_angle) / len(ranges)) * target_index - self.lidar_angle
                
        #         # future_pos_index = (curr_pos_index + 20) % l
        #         # goal_point = ref[future_pos_index]
        #         # print(goal_point)
        #         # curr_pos = copy.copy(self.pos)  # [x, y, yaw, speed]
        #         # dx = goal_point[0] - curr_pos[0]
        #         # dy = goal_point[1] - curr_pos[1]
        #         # yaw = curr_pos[2]
        #         # xgv = (dx * np.cos(yaw)) + (dy * np.sin(yaw))
        #         # ygv = (-dx * np.sin(yaw)) + (dy * np.cos(yaw))
        #         # angle = -np.arctan2(ygv,xgv)
        #         # print(angle)
        #         self.step(angle)
        #         curr_pos_index, _ = self.__find_closest_point(curr_pos_index)
            
        #     self.prev_pos_index = copy.copy(curr_pos_index)

        
        # print(self.prev_pos_index)

        return


if __name__ == '__main__':
    file_name = '~/rl_library/applications/f1_10_simplex/porto_waypoints.csv'
    # df = pd.read_csv(file_name, names=['X', 'Y', 'Z?', 'W'])
    # waypoints = df[['X', 'Y']].to_numpy()
    # print(waypoints)
    # fig, ax = plt.subplots()
    # ax.plot(waypoints[:, 0], waypoints[:, 1])
    # # plt.show()
    # ref_track = np.zeros((len(waypoints), 3), dtype=float)
    # ref_track[0, :] = [waypoints[0, 0], waypoints[0, 1], 0.0]
    # tot_dist = 0.0
    # for i in range(1, len(waypoints)):
    #     dist = np.sqrt((waypoints[i, 0] - waypoints[(i - 1), 0])**2 + (waypoints[i, 1] - waypoints[(i - 1), 1])**2)
    #     tot_dist += dist
    #     ref_track[i, :] = [waypoints[i, 0], waypoints[i, 1], tot_dist]
    #
    # print(ref_track)
    # run = F110Runner(file_name, 'p', 'p', 'p')
    # run.pos = run.ref_track[5, :] + 0.1*np.random.rand(3)
    # print(run.pos)
    # r = run.find_closest_point(1977)
    # print(r)
    # print(run.ref_track[r, :])
    run = F110Runner(file_name, 'p', 'p', 'p')
