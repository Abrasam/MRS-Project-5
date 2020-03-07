#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

import scipy.special

from divide_areas import divide
import time


# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion


def braitenberg(front, front_left, front_right, left, right):
    u = 0.  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.

    # MISSING: Implement a braitenberg controller that takes the range
    # measurements given in argument to steer the robot.

    t_front = np.tanh(front / 2)
    t_front_left = np.tanh(front_left / 2)
    t_front_right = np.tanh(front_right / 2)
    t_left = np.tanh(left / 2)
    t_right = np.tanh(right / 2)

    u = (t_front - 0.1)
    w = 0.6 * (t_front_left - t_front_right) + 0.3 * (t_left - t_right)

    return u, w


def rule_based(front, front_left, front_right, left, right):
    u = 0.25  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.

    if front < 0.25:
        u = 0
        w = -0.25
    if front_left < 0.2:
        u = 0
        w = -0.2
    elif front_right < 0.2:
        u = 0
        w = 0.2
    elif left < 0.15:
        w = -0.2
    elif right < 0.15:
        w = 0.2
    return u, w


class SimpleLaser(object):
    def __init__(self, name):
        rospy.Subscriber('/' + name + '/scan', LaserScan, self.callback)
        self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
        self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
        self._measurements = [float('inf')] * len(self._angles)
        self._indices = None

    def callback(self, msg):
        # Helper for angles.
        def _within(x, a, b):
            pi2 = np.pi * 2.
            x %= pi2
            a %= pi2
            b %= pi2
            if a < b:
                return a <= x and x <= b
            return a <= x or x <= b

        # Compute indices the first time.
        if self._indices is None:
            self._indices = [[] for _ in range(len(self._angles))]
            for i, d in enumerate(msg.ranges):
                angle = msg.angle_min + i * msg.angle_increment
                for j, center_angle in enumerate(self._angles):
                    if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
                        self._indices[j].append(i)

        ranges = np.array(msg.ranges)
        for i, idx in enumerate(self._indices):
            # We do not take the minimum range of the cone but the 10-th percentile for robustness.
            self._measurements[i] = np.percentile(ranges[idx], 10)

    @property
    def ready(self):
        return not np.isnan(self._measurements[0])

    @property
    def measurements(self):
        return self._measurements


class GroundtruthPose(object):
    def __init__(self, name='tb3_0'):
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._name = name

    def callback(self, msg):
        idx = [i for i, n in enumerate(msg.name) if n == self._name]
        if not idx:
            raise ValueError(
                'Specified name "{}" does not exist.'.format(self._name))
        idx = idx[0]
        self._pose[0] = msg.pose[idx].position.x
        self._pose[1] = msg.pose[idx].position.y
        _, _, yaw = euler_from_quaternion([
            msg.pose[idx].orientation.x,
            msg.pose[idx].orientation.y,
            msg.pose[idx].orientation.z,
            msg.pose[idx].orientation.w])
        self._pose[2] = yaw

    @property
    def ready(self):
        return not np.isnan(self._pose[0])

    @property
    def pose(self):
        return self._pose


def run(args):
    rospy.init_node('full_coverage')
    avoidance_method = globals()[args.mode]

    # Update control every 100 ms.
    rate_limiter = rospy.Rate(100)
    publishers = []
    lasers = []
    ground_truths = []
    pose_histories = []
    for robot in ["tb3_0", "tb3_1", "tb3_2"]:
        publishers.append(rospy.Publisher(
            '/' + robot + '/cmd_vel', Twist, queue_size=5))
        lasers.append(SimpleLaser(name=robot))
        # Keep track of groundtruth position for plotting purposes.
        ground_truths.append(GroundtruthPose(name=robot))
        pose_histories.append([])
    with open('/tmp/gazebo_exercise.txt', 'w'):
        pass

    start_timer = time.time()
    while not rospy.is_shutdown():
        # Make sure all measurements are ready.
        if not all(laser.ready for laser in lasers) or not all(groundtruth.ready for groundtruth in ground_truths):
            rate_limiter.sleep()
            start_timer = time.time()
            continue

        if time.time() - start_timer < 5: # Run around for 10 seconds
            for index, robot in enumerate(["tb3_0", "tb3_1", "tb3_2"]):
                u, w = avoidance_method(*lasers[index].measurements)
                vel_msg = Twist()
                vel_msg.linear.x = u
                vel_msg.angular.z = w
                publishers[index].publish(vel_msg)

                # Log groundtruth positions in /tmp/gazebo_exercise.txt
                pose_histories[index].append(ground_truths[index].pose)
                if len(pose_histories[index]) % 10:
                    with open('/tmp/gazebo_robot_' + robot + '.txt', 'a') as fp:
                        #fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
                        pose_histories[index] = []
            rate_limiter.sleep()
            continue
        # Stop all Robots
        u, w = 0, 0
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        for index, robot in enumerate(["tb3_0", "tb3_1", "tb3_2"]):
            publishers[index].publish(vel_msg)
            # Log groundtruth positions in /tmp/gazebo_exercise.txt
            pose_histories[index].append(ground_truths[index].pose)
            if len(pose_histories[index]) % 10:
                with open('/tmp/gazebo_robot_' + robot + '.txt', 'a') as fp:
                    #fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
                    pose_histories[index] = []

        # Locations - currenlty use ground truth
        # TODO - must switch to localization result

        robot_locations = [(i.pose[0] , i.pose[1]) for i in ground_truths]
        print(robot_locations)
        movement_functions = divide(args, robot_locations, 450)
        print([i(0) for i in movement_functions])
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs obstacle avoidance')
    parser.add_argument('--mode', action='store', default='braitenberg',
                        help='Method.', choices=['braitenberg', 'rule_based'])
    #parser.add_argument('--robot', action='store')
    parser.add_argument('--map', action='store', default='../ros/world_map',
                        help='Which map to use.')
    args, unknown = parser.parse_known_args()
    try:
        run(args)
    except rospy.ROSInterruptException:
        pass
