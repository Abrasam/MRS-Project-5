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

import matplotlib.pylab as plt

EPSILON = .1
NUMBER_ROBOTS = 1
ROBOT_SPEED = 0.1


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


  u = velocity[0]*np.cos(pose[2]) + velocity[1]*np.sin(pose[2])
  w = (velocity[1]*np.cos(pose[2]) - velocity[0]*np.sin(pose[2])) / epsilon

  return u, w

def get_velocity(position, target, robot_speed):

  v = np.zeros_like(position)
  position[0] += EPSILON*np.cos(position[2])
  position[1] += EPSILON*np.sin(position[2])
  #
  target_vel = np.array([robot_speed*np.cos(target[2]), robot_speed*np.sin(target[2]), 0])

  # Head towards the next point
  v = (target - position)
  v /= np.linalg.norm(v)
  v /= 2
  v += target_vel
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


def run(args):
    rospy.init_node('full_coverage')
    avoidance_method = globals()[args.mode]

    # Update control every 100 ms.
    rate_limiter = rospy.Rate(10)
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

    # plotting values
    times = []
    trajectory = [[], []]
    poses = [[], []]
    counter = 0

    start_timer = time.time()
    paths_found = False
    run_time_started = False
    while not rospy.is_shutdown():
        # Make sure all measurements are ready.
        if not all(laser.ready for laser in lasers) or not all(groundtruth.ready for groundtruth in ground_truths):
            rate_limiter.sleep()
            start_timer = time.time()
            continue

        if time.time() - start_timer < 2: # Run around for 10 seconds
            for index in range(NUMBER_ROBOTS):
                robot = "tb3_%s" % index
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
                pose_histories[index].append(ground_truths[index].pose)
                if len(pose_histories[index]) % 10:
                    with open('/tmp/gazebo_robot_' + robot + '.txt', 'a') as fp:
                        #fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
                        pose_histories[index] = []

            # Locations - currenlty use ground truth
            # TODO - must switch to localization result
            time.sleep(1)
            # Transposing location
            robot_locations = [(i.pose[0] , i.pose[1]) for i in ground_truths]
            print(robot_locations)
            movement_functions = divide(args, robot_locations[:NUMBER_ROBOTS], ROBOT_SPEED)
            if movement_functions == False:
                time.sleep(2)
                start_time = time.time()
                print(robot_locations)
                print("Fail")
                continue
            paths_found = True
            print(robot_locations)
            for i in ground_truths:
                print(i.pose)
            print()
            for i in movement_functions:
                print(i(0))

        # Follow path
        if not run_time_started:
            run_time_started = True
            run_time = time.time()
        for index in range(NUMBER_ROBOTS):
            robot = "tb3_%s" % index
            target = movement_functions[index](time.time() - run_time)
            v = get_velocity(ground_truths[index].pose.copy(), target, ROBOT_SPEED)

            u, w = feedback_linearized(ground_truths[index].pose.copy(), v, epsilon=EPSILON)
            print("%.2f, %.2f, %.2f -- %.2f, %.2f, %.2f     u:%.2f, w:%.2f" % (ground_truths[index].pose[0], ground_truths[index].pose[1], ground_truths[index].pose[2], target[0], target[1], target[2], u, w))
            times.append(time.time())
            trajectory[0].append(target[0])
            trajectory[1].append(target[1])
            poses[0].append(ground_truths[index].pose[0])
            poses[1].append(ground_truths[index].pose[1])
            counter += 1
            vel_msg = Twist()
            vel_msg.linear.x = u
            vel_msg.angular.z = w
            publishers[index].publish(vel_msg)
        if counter % 100000 == 0:
            fig = plt.figure()
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim([-4, 4])
            plt.ylim([-4, 4])

            plt.scatter(trajectory[0], trajectory[1], c = 'b', linewidths=0, edgecolors='face')
            plt.scatter(poses[0], poses[1], c = 'r', linewidths=0, edgecolors='face')

            run_time_pause = time.time() - run_time
            plt.show()
            run_time = time.time()-run_time_pause


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
