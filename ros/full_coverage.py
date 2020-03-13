#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

import scipy.special

from divide_areas import divide, create_occupancy_grid
from plot_trajectory_nav import plot_trajectory
import time

from copy import deepcopy
import os

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist, Point32
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

import matplotlib.pylab as plt


from threading import Thread

NUMBER_ROBOTS = 1
ROBOT_SPEED = 0.2

ROBOT_RADIUS = 0.105 / 2.
EPSILON = ROBOT_RADIUS


def rule_based(front, front_left, front_right, left, right):
    u = 0.25  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.

    if front < 0.25:
        u = 0
        w = -0.5
    if front_left < 0.2:
        u = 0
        w = -0.5
    elif front_right < 0.2:
        u = 0
        w = 0.5
    elif left < 0.15:
        w = -0.5
    elif right < 0.15:
        w = 0.5
    return u, w


def feedback_linearized(pose, velocity, epsilon):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # Implement feedback-linearization to follow the velocity
  # vector given as argument. Epsilon corresponds to the distance of
  # linearized point in front of the robot.

  # print("velocity", velocity)
  u = velocity[0] * np.cos(pose[2]) + velocity[1] * np.sin(pose[2])
  w = (velocity[1] * np.cos(pose[2]) - velocity[0] * np.sin(pose[2])) / epsilon

  # u = velocity[0]*np.cos(pose[2]) + velocity[1]*np.sin(pose[2])
  # w = velocity[2]
  return u, w


def get_velocity(position, target, robot_speed, expected_direction=None):

  v = (target - position)
  v /= np.linalg.norm(v[:2])
  v /= 4
  return v

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

    def apply_motion_model(self, u, w, dt):
        pass

class LocalisationPose(object):
    def __init__(self, name='tb3_0'):
        rospy.Subscriber('/locpos' + name[-1], Point32, self.callback)
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._name = name
        self.prediction_publisher = rospy.Publisher('/loc_motion_model' + name[-1], PointCloud, queue_size=1)
        self.frame_id = 0

    def callback(self, msg):
        self._pose[0] = msg.x
        self._pose[1] = msg.y
        self._pose[2] = ((msg.z + np.pi) % (2 * np.pi)) - np.pi
        # print("YAW                    ", self._pose[2])

    def apply_motion_model(self, u, w, dt):
        vel_x = u * np.cos(self._pose[2])
        vel_y = u * np.sin(self._pose[2])
        vel_theta = w
        self._pose[0] += vel_x * dt
        self._pose[1] += vel_y * dt
        self._pose[2] += vel_theta * dt


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
    refresh_Hz = 100
    loop_time = 1.0 / refresh_Hz
    rate_limiter = rospy.Rate(refresh_Hz)
    publishers = []
    lasers = []
    estimated_positions = []
    ground_truths = []
    pose_history = []
    for index in range(NUMBER_ROBOTS):
        robot = "tb3_" + str(index)
        publishers.append(rospy.Publisher(
            '/' + robot + '/cmd_vel', Twist, queue_size=5))
        lasers.append(SimpleLaser(name=robot))
        # Keep track of groundtruth position for plotting purposes.
        ground_truths.append(GroundtruthPose(name=robot))
        #estimated_positions.append(LocalisationPose(name=robot))
        estimated_positions.append(GroundtruthPose(name=robot))
        pose_history.append([])

    # plotting values
    times = []
    for i in range(NUMBER_ROBOTS):
      with open('/tmp/gazebo_robot_nav_tb3_' + str(i) + '.txt', 'w'):
        pass
    counter = 0

    targets = [0] * NUMBER_ROBOTS
    arrived = [False] * NUMBER_ROBOTS

    start_timer = time.time()
    paths_found = False
    run_time_started = False
    counter = 0

    covered_locations = []
    # Initialise map to cover.
    original_occupancy_grid, _, _ = create_occupancy_grid(args)
    cover_grid, _, _ = create_occupancy_grid(args)

    total_to_cover = original_occupancy_grid.total_free()


    while not rospy.is_shutdown():
        # Make sure all measurements are ready.
        if not all(laser.ready for laser in lasers) or not all(groundtruth.ready for groundtruth in estimated_positions):
            rate_limiter.sleep()
            start_timer = time.time()
            continue

        # Use separate list for all covered locations - don't split by robot - orientation doesn't matter
        """for i in ground_truths:
            x, y = i.pose[:2]
            res = original_occupancy_grid.resolution
            #print(res, 2.0*ROBOT_RADIUS/res)
            #print("xy", x,y)
            for a in np.linspace(x - ROBOT_RADIUS, x + ROBOT_RADIUS,20):
                for b in np.linspace(y - ROBOT_RADIUS, y + ROBOT_RADIUS, 20):
                    #print(a, b, ((a - x)**2 + (b - y)**2) < ROBOT_RADIUS**2)
                    if ((a - x)**2 + (b - y)**2) <= ROBOT_RADIUS**2:
                        #covered_locations.append((a, b))
                        if original_occupancy_grid.is_free((a, b)):
                            if cover_grid.is_free((a, b)):
                                w, z = cover_grid.get_index((a, b))
                                cover_grid.values[w, z] = 3



        total_covered = np.sum(cover_grid.values == 3)
        print(total_covered, total_to_cover)"""


        while not os.path.exists("/go"):
            for index in range(NUMBER_ROBOTS):
                robot = "tb3_%s" % index
                u, w = avoidance_method(*lasers[index].measurements)
                vel_msg = Twist()
                vel_msg.linear.x = u
                vel_msg.angular.z = w
                publishers[index].publish(vel_msg)
                estimated_positions[index].apply_motion_model(u, w, loop_time)
            """for i in ground_truths:
                x, y = i.pose[:2]
                res = original_occupancy_grid.resolution
                #print("xy", x,y)
                for a in np.linspace(x - ROBOT_RADIUS, x + ROBOT_RADIUS, 30):
                    for b in np.linspace(y - ROBOT_RADIUS, y + ROBOT_RADIUS, 30):
                        #print(a, b, ((a - x)**2 + (b - y)**2) < ROBOT_RADIUS**2)
                        if ((a - x)**2 + (b - y)**2) <= ROBOT_RADIUS**2:
                            #covered_locations.append((a, b))
                            if original_occupancy_grid.is_free((a, b)):
                                if cover_grid.is_free((a, b)):
                                    w, z = cover_grid.get_index((a, b))
                                    cover_grid.values[w, z] = 3"""
            rate_limiter.sleep()
            #print(covered_locations[-10:])

        if not paths_found:
            # Stop all Robots
            u, w = 0, 0
            vel_msg = Twist()
            vel_msg.linear.x = u
            vel_msg.angular.z = w
            for index in range(NUMBER_ROBOTS):
                robot = "tb3_%s" % index
                publishers[index].publish(vel_msg)

            # Locations - currenlty use ground truth
            # TODO - must switch to localization result
            # time.sleep(1)
            # Transposing location
            robot_locations= [(i.pose[0], i.pose[1]) for i in estimated_positions]
            print(robot_locations)
            robot_paths = divide(args, robot_locations[:NUMBER_ROBOTS], ROBOT_SPEED)
            if robot_paths == False:
                time.sleep(2)
                start_time = time.time()
                print(robot_locations)
                print("Fail")
                continue
            paths_found = True
            print(robot_locations)
            for i in estimated_positions:
                print(i.pose)
            print()
            for i in robot_paths:
                print(i[0])

        # Follow path
        if not run_time_started:
            run_time_started = True
            run_time = time.time()
        for index in range(NUMBER_ROBOTS):
            robot = "tb3_%s" % index

            current_target = robot_paths[index][targets[index]]
            current_position = estimated_positions[index].pose.copy()
            # Check if at target.
            distance = ((current_target[0] - current_position[0]) ** 2
                     + (current_target[1] - current_position[1]) ** 2) ** 0.5

            if distance < 1 * ROBOT_RADIUS or arrived[index]:
                # Keep moving for a bit
                arrived[index] = True
                # Within 3 degrees
                if np.absolute((current_target[2]) - current_position[2]) < (0.1):
                    """if index == 0:
                        print("Next")"""
                    arrived[index] = False
                    targets[index] += 1
                    targets[index] %= len(robot_paths[index])
                    current_target = robot_paths[index][targets[index]]
                    # print(current_target)
                    v = get_velocity(current_position.copy(), deepcopy(current_target), ROBOT_SPEED, expected_direction=robot_paths[index][targets[index] - 1][2])
                    # v = np.array([1, 0])
                    u, w = feedback_linearized(current_position.copy(), v, epsilon=EPSILON)
                    # u=0.5
                    # w=0
                else:
                    """if index == 0:
                        print("Rotating")"""
                    # Rotate to correct orientation
                    u = 0
                    difference = ((current_target[2] % (2 * np.pi)) - (current_position[2] % (2 * np.pi))) % (2 * np.pi)

                    if difference < np.pi:
                        # Difference heading to 0
                        # w = max(0.25, difference
                        w = 0.25
                        w = 1
                    else:
                        remaining = 2 * np.pi - difference
                        # w = -1*max(0.25, remaining)
                        w = -0.25
                        w = -1
                    # w = 0.2 if ((current_target[2]) - current_position[2]) > 0 and (current_target[2] - current_position[2]) < np.pi else -0.2
            else:
                """if index == 0:
                    print("Moving")"""
                v = get_velocity(deepcopy(current_position), deepcopy(current_target), ROBOT_SPEED, expected_direction=robot_paths[index][targets[index] - 1][2])
                # v = np.array([1, 0])
                u, w = feedback_linearized(deepcopy(current_position), v, epsilon=EPSILON)
                #w = np.clip(w, -0.3, 0.3)
                # u = 0.5
                # w = 0
            vel_msg = Twist()
            vel_msg.linear.x = u
            vel_msg.angular.z = w
            publishers[index].publish(vel_msg)
            estimated_positions[index].apply_motion_model(u, w, loop_time)

            pose_history[index].append(ground_truths[index].pose)
            if len(pose_history[index]) % 10:
              with open('/tmp/gazebo_robot_nav_tb3_' + str(index) + '.txt', 'a') as fp:
                fp.write('\n'.join(','.join(str(v) for v in p)
                         for p in pose_history[index])  + (",%s" % (time.time() - run_time)) + '\n')
                pose_history[index] = []
        #print(covered_locations[-10:])
        rate_limiter.sleep()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs obstacle avoidance')
    parser.add_argument('--mode', action='store', default='braitenberg',
                        help='Method.', choices=['braitenberg', 'rule_based'])
    # parser.add_argument('--robot', action='store')
    parser.add_argument('--map', action='store', default='../ros/world_map',
                        help='Which map to use.')
    args, unknown = parser.parse_known_args()
    try:
        run(args)
    except rospy.ROSInterruptException:
        pass
