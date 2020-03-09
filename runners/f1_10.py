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
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from race.msg import drive_param

from runners.abstract_runner import Runner


class F110Runner(Runner):
    def __init__(self, reference_file_name, lidar_sub_name, odom_sub_name, control_pub_name, rate=10,
                 min_action=np.asarray([1.0, (-34.0 * np.pi / 180.)]),
                 max_action=np.asarray([3.0, (34.0 * np.pi / 180.)]), scale=0,
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
        self.lidar_angle = lidar_angle
        self.max_lidar_range = max_lidar_range
        self.min_dist = min_dist
        self.crash_threshold = crash_threshold

        self.scale_mult = (max_action - min_action) / 2.0
        self.scale_add = (max_action - min_action) / 2.0 + min_action

        self.indices = None

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

    def __calculate_reward(self):
        """
        TODO: write up explanation
        :return:
        """

        # Determine the current position index
        curr_pos_index = self.__find_closest_point(self.prev_pos_index)

        # Calculate the distance traveled between points
        reward = self.ref_track[curr_pos_index, 2] - self.ref_track[self.prev_pos_index, 2]

        # The reward will be a very large negative value if a lap was completed, so recompute in those cases
        if reward < -5.0:
            end = len(self.ref_track) - 1
            dist_to_end = self.ref_track[end, 2] - self.ref_track[self.prev_pos_index, 2]
            dist_from_start = self.ref_track[curr_pos_index, 2]
            reward = dist_to_end + dist_from_start

        self.prev_pos_index = curr_pos_index

        return reward

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

        self.pos = np.asarray([x, y, yaw, speed])
        # print('ego_pos updated')

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

            # Compute the indices of values within +/- the desired lidar angle
            self.indices = np.where(((-1 * self.lidar_angle) <= angles) & (angles <= self.lidar_angle))
            self.obs_shape = self.indices.shape

        # Clip any range values larger than the set maximum since the lidar data is very noisy at larger distances
        max_ranges = self.max_lidar_range * np.ones_like(ranges)
        clipped_ranges = np.maximum(ranges, max_ranges)

        self.lidar_ranges = clipped_ranges

        return

    def __find_closest_point(self, prev_pos):
        """
        TODO: write explanation
        :param prev_pos:
        :return:
        """

        # Record the current position
        curr_pos = self.pos
        ref = self.ref_track
        l = len(ref)

        # Find the closest point using a binary search-esque method
        index_min = (prev_pos - 10) % l
        index_max = (prev_pos + 20) % l
        select_index = index_min
        min_dist = 100.0
        iters = 0
        while iters < 10:
            if index_max > index_min:
                for i in range(index_min, index_max):
                    dist = np.sqrt((ref[i, 0] - curr_pos[0])**2 + (ref[i, 1] - curr_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        select_index = i
            else:
                for i in range(index_min, l):
                    dist = np.sqrt((ref[i, 0] - curr_pos[0]) ** 2 + (ref[i, 1] - curr_pos[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        select_index = i
                for i in range(index_max):
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

        return select_index

    def get_state(self):
        """
        This function returns the current state of the agent in the environment.

        :return curr_state: (np.array) The current state of the agent is the selected lidar values.
        """
        curr_state = self.lidar_ranges[self.indices]

        return curr_state

    def is_available(self):
        """
        This method checks to make sure the environment is still available. Some environments are able to discontinue
        without stopping the learning process.

        :input:
            None
        :output:
            return 0/1 (0 if unavailable, 1 if available)
        """

        return not rospy.is_shutdown()

    def __is_terminal(self):
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
        indices = np.where(self.lidar_ranges <= self.min_dist)
        if len(indices) >= self.crash_threshold:
            exit_cond = 1

        return done, exit_cond

    def __load_reference_track(self, file_name):
        """
        TODO: documentation
        :param file_name:
        :return:
        """

        # Read the waypoint file
        df = pd.read_csv(file_name, names=['X', 'Y', 'Z', 'W'])
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

    def step(self, action):
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
        vel_cmd = action[0]
        steer_cmd = action[1]
        self.__publish_cmd(vel_cmd, steer_cmd)

        # Wait specified time
        self.rate.sleep()

        # Collect new state
        next_state = self.get_state()
        reward = self.__calculate_reward()
        done, exit_cond = self.__is_terminal()

        return next_state, reward, done, exit_cond

    def stop(self):
        """
        This method stops the car.
        """
        self.__publish_cmd(0.0, 0.0)

        return

    def reset(self):
        """
        This function resets the simulation environment.
        """

        self.reset_env()
        self.rate.sleep()

        return


if __name__ == '__main__':
    file_name = '~/Platooning-F1Tenth/src/a_stars_pure_pursuit/waypoints/waypoints.csv'
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
