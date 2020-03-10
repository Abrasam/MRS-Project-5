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

from copy import deepcopy
import os

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist,Point32
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan, PointCloud
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

import matplotlib.pylab as plt


NUMBER_ROBOTS = 3
ROBOT_SPEED = 0.3

ROBOT_RADIUS = 0.105 / 2.
EPSILON = ROBOT_RADIUS

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

def feedback_linearized(pose, velocity, epsilon):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # Implement feedback-linearization to follow the velocity
  # vector given as argument. Epsilon corresponds to the distance of
  # linearized point in front of the robot.

  #print("velocity", velocity)
  u = velocity[0]*np.cos(pose[2]) + velocity[1]*np.sin(pose[2])
  w = (velocity[1]*np.cos(pose[2]) - velocity[0]*np.sin(pose[2])) / epsilon

  #u = velocity[0]*np.cos(pose[2]) + velocity[1]*np.sin(pose[2])
  #w = velocity[2]
  return u, w

def get_velocity(position, target, robot_speed):

  v = np.zeros_like(position)
  #position[0] += EPSILON*np.cos(position[2])
  #position[1] += EPSILON*np.sin(position[2])
  #
  #target_vel = np.array([robot_speed*np.cos(target[2]), robot_speed*np.sin(target[2]), 0])

  # Head towards the next point
  v = (target - position)
  v /= np.linalg.norm(v[:2])
  v /= 3
  #v += target_vel
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

class LocalisationPose(object):
    def __init__(self, name='tb3_0'):
        rospy.Subscriber('/locpos'+name[-1], Point32, self.callback)
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._name = name
        self.prediction_publisher = rospy.Publisher('/motion_model_pred' + name[-1], PointCloud, queue_size=1)
        self.frame_id = 0

    def callback(self, msg):
        self._pose[0] = msg.x
        self._pose[1] = msg.y
        self._pose[2] = msg.z

    def apply_motion_model(self, u, w, dt):
        vel_x = u * np.cos(self._pose[2])
        vel_y = u * np.sin(self._pose[2])
        vel_theta = w
        self.pose[0] += vel_x * dt
        self.pose[1] += vel_y * dt
        self.pose[2] += vel_theta * dt

        msg = PointCloud()
        msg.header.seq = self.frame_id
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = '/tb3_'+str(self._name[-1])+'/pred'
        pt = Point32()
        pt.x = self.pose[0]
        pt.y = self.pose[1]
        pt.z = self.pose[2]
        msg.points.append(pt)
        #msg.header.frame_id = '/'+str(self._name)+'/pred'
        self.prediction_publisher.publish(msg)
        self.frame_id += 1

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
    for robot in ["tb3_0", "tb3_1", "tb3_2"]:
        publishers.append(rospy.Publisher(
            '/' + robot + '/cmd_vel', Twist, queue_size=5))
        lasers.append(SimpleLaser(name=robot))
        # Keep track of groundtruth position for plotting purposes.
        ground_truths.append(GroundtruthPose(name=robot))
        estimated_positions.append(LocalisationPose(name=robot))
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
    while not rospy.is_shutdown():
        # Make sure all measurements are ready.
        if not all(laser.ready for laser in lasers) or not all(groundtruth.ready for groundtruth in estimated_positions):
            rate_limiter.sleep()
            start_timer = time.time()
            continue

        #print(os.getcwd())
        #if time.time() - start_timer < 2: # Run around for 10 seconds

        while not os.path.exists("/go"):
            for index in range(NUMBER_ROBOTS):
                robot = "tb3_%s" % index
                u, w = avoidance_method(*lasers[index].measurements)
                vel_msg = Twist()
                vel_msg.linear.x = u
                vel_msg.angular.z = w
                publishers[index].publish(vel_msg)

                """# Log groundtruth positions in /tmp/gazebo_exercise.txt
                pose_histories[index].append(estimated_positions[index].pose)
                if len(pose_histories[index]) % 10:
                    with open('/tmp/gazebo_robot_' + robot + '.txt', 'a') as fp:
                        #fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
                        pose_histories[index] = []"""
            rate_limiter.sleep()
            continue

        if not paths_found:
            # Stop all Robots
            u, w = 0, 0
            vel_msg = Twist()
            vel_msg.linear.x = u
            vel_msg.angular.z = w
            for index in range(NUMBER_ROBOTS):
                robot = "tb3_%s" % index
                publishers[index].publish(vel_msg)
                # Log groundtruth positions in /tmp/gazebo_exercise.txt
                """pose_histories[index].append(estimated_positions[index].pose)
                if len(pose_histories[index]) % 10:
                    with open('/tmp/gazebo_robot_' + robot + '.txt', 'a') as fp:
                        #fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
                        pose_histories[index] = []"""

            # Locations - currenlty use ground truth
            # TODO - must switch to localization result
            #time.sleep(1)
            # Transposing location
            robot_locations = [(i.pose[0] , i.pose[1]) for i in estimated_positions]
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
                     +  (current_target[1] - current_position[1]) ** 2) ** 0.5

            if distance < 4*ROBOT_RADIUS or arrived[index]:
                # Keep moving for a bit
                arrived[index] = True
                if np.absolute((current_target[2])-current_position[2]) < (0.2): # Within 3 degrees
                    #print("Next")
                    arrived[index] = False
                    targets[index] += 1
                    targets[index] %= len(robot_paths[index])
                    current_target = robot_paths[index][targets[index]]
                    #print(current_target)
                    v = get_velocity(current_position.copy(), deepcopy(current_target), ROBOT_SPEED)
                    #v = np.array([1, 0])
                    u, w = feedback_linearized(current_position.copy(), v, epsilon=EPSILON)
                    #u=0.5
                    #w=0
                else:
                    #print("Rotating")
                    # Rotate to correct orientation
                    u = 0
                    difference = ((current_target[2]%(2*np.pi)) - (current_position[2]%(2*np.pi)))%(2*np.pi)

                    if difference < np.pi:
                        # Difference heading to 0
                        w = max(0.75, difference)
                    else:
                        remaining = 2*np.pi - difference
                        w = -1*max(0.75, remaining)
                    #w = 0.2 if ((current_target[2]) - current_position[2]) > 0 and (current_target[2] - current_position[2]) < np.pi else -0.2
            else:
                #print("Moving")
                v = get_velocity(deepcopy(current_position), deepcopy(current_target), ROBOT_SPEED)
                #v = np.array([1, 0])
                u, w = feedback_linearized(deepcopy(current_position), v, epsilon=EPSILON)
                #u = 0.5
                #w = 0

            #print("%.2f, %.2f, %.2f -- %.2f, %.2f, %.2f     u:%.2f, w:%.2f" % (current_position[0], current_position[1], current_position[2], current_target[0], current_target[1], current_target[2], u, w))

            vel_msg = Twist()
            vel_msg.linear.x = u
            vel_msg.angular.z = w
            publishers[index].publish(vel_msg)
            estimated_positions[index].apply_motion_model(u, w, loop_time)

            pose_history[index].append(ground_truths[index].pose)
            if len(pose_history[index]) % 10:
              with open('/tmp/gazebo_robot_nav_tb3_' + str(index) + '.txt', 'a') as fp:
                fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history[index]) + '\n')
                pose_history[index] = []

        rate_limiter.sleep()


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
