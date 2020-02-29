#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import numpy as np
import rospy
import time

import scipy.special
import scipy.stats

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
# For displaying particles.
# http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud.html
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
# Odometry.
from nav_msgs.msg import Odometry

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

ROBOT_RADIUS = 0.105 / 2.
WALL_OFFSET = 2.
CYLINDER_POSITION = np.array([.3, .2], dtype=np.float32)
CYLINDER_RADIUS = .3 + ROBOT_RADIUS


def braitenberg(front, front_left, front_right, left, right):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # MISSING: Implement a braitenberg controller that takes the range
  # measurements given in argument to steer the robot safely.

  t_front = np.tanh(front / 2)
  t_front_left = np.tanh(front_left / 2)
  t_front_right = np.tanh(front_right / 2)
  t_left = np.tanh(left / 2)
  t_right = np.tanh(right / 2)

  u = 1.4 * (t_front - 0.1)
  w = 0.6 * (t_front_left - t_front_right) + 0.3 * (t_left - t_right) - 0.1 * t_front

  return u, w


class Particle(object):
  """Represents a particle."""

  def __init__(self):
    self._pose = np.zeros(3, dtype=np.float32)
    self._weight = 1.

    # MISSING: Initialize a particle randomly in the arena. Set the values of
    # _pose such that it is a valid pose (i.e., inside the arena walls, but
    # outside the cylinder). Consider implementing is_valid() below and use it
    # in this function.

    self.randomize()

  def randomize(self):
    valid = False

    while not valid:
      self._pose = np.random.rand(3)
      self._pose[X] = (self._pose[X] - 0.5) * 4.0
      self._pose[Y] = (self._pose[Y] - 0.5) * 4.0
      self._pose[YAW] *= 2 * np.pi

      valid = self.is_valid()

    self._weight = 1.0

  def is_valid(self):
    # MISSING: Implement a function that returns True if the current particle
    # position is valid. You might need to use this function in __init__()
    # and compute_weight().

    if np.abs(self._pose[X]) > WALL_OFFSET - ROBOT_RADIUS:
      return False
    elif np.abs(self._pose[Y]) > WALL_OFFSET - ROBOT_RADIUS:
      return False

    pos = np.array([self._pose[X], self._pose[Y]], dtype=np.float32)

    cyl_offset = np.subtract(pos, CYLINDER_POSITION)
    cyl_dist = np.linalg.norm(cyl_offset)

    if cyl_dist < CYLINDER_RADIUS + ROBOT_RADIUS:
      return False

    return True

  def move(self, delta_pose):
    # MISSING: Update the particle pose according the motion model.
    # delta_pose is an offset in the particle frame. As motion model,
    # use roughtly 10% standard deviation with respect to the forward
    # and rotational velocity.
    #
    # In a second step, make the necessary modifications to handle the
    # kidnapped robot problem. For example, with a low probability the
    # particle can be repositioned randomly in the arena.

    u = delta_pose[X]
    w = delta_pose[YAW]

    u *= np.random.normal(loc=1.0, scale=0.1)
    w *= np.random.normal(loc=1.0, scale=0.1)

    self._pose[X] += u * np.cos(self._pose[YAW] + 0.5 * w)
    self._pose[Y] += u * np.sin(self._pose[YAW] + 0.5 * w)
    self._pose[YAW] += w

    if np.random.rand() < 0.01 \
        or (1.4 < self._weight < 1.8 and np.random.rand() < 0.2) \
        or (1.8 < self._weight < 2.2 and np.random.rand() < 0.1):

      # rotate around origin
      rot = np.random.rand() * 2 * np.pi
      c, s = np.cos(rot), np.sin(rot)
      nx = self._pose[X] * c + self._pose[Y] * (-s)
      ny = self._pose[X] * s + self._pose[Y] * c
      self._pose[X] = nx
      self._pose[Y] = ny
      self._pose[YAW] += rot
      if self._pose[YAW] > 2 * np.pi:
        self._pose[YAW] -= np.pi
    elif np.random.rand() < 0.01 or (self._weight < 1.4 and np.random.rand() < 0.05):
      self.randomize()

    # if delta_pose was X,Y,YAW rather than FORWARD,0,YAW
    # np.add(self._pose, np.multiply(delta_pose, np.random.normal(loc=1.0, scale=0.1, size=3)), self._pose)

    # print(delta_pose)

    pass

  def compute_weight(self, front, front_left, front_right, left, right):
    # MISSING: Update the particle weight self._weight according to measurements.
    # You can use the self.ray_trace(angle) function below. Remember to reduce the
    # weight of particles that are outside the arena. As measurement model, use a
    # Gaussian error with a standard deviation of 80 [cm]. Note that the maximum
    # range of the laser-range finder is 3.5 meters (observations beyond this range
    # will show up as infinity).
    sigma = .8
    variance = sigma ** 2.

    inf_replace = 3.5

    front = np.minimum(front, inf_replace)
    front_left = np.minimum(front_left, inf_replace)
    front_right = np.minimum(front_right, inf_replace)
    left = np.minimum(left, inf_replace)
    right = np.minimum(right, inf_replace)

    # 1 - probability of being between -x and +x in the normal distribution with std dev of sigma
    def erfc(x):
      return 1.0 - scipy.special.erf(x / (sigma * np.sqrt(2)))

    def angle_erfc(angle):
      pred_front = np.minimum(self.ray_trace(angle), inf_replace)
      pred_front_left = np.minimum(self.ray_trace(angle + np.pi / 4.0), inf_replace)
      pred_front_right = np.minimum(self.ray_trace(angle - np.pi / 4.0), inf_replace)
      pred_left = np.minimum(self.ray_trace(angle + np.pi / 2.0), inf_replace)
      pred_right = np.minimum(self.ray_trace(angle - np.pi / 2.0), inf_replace)

      diff_front = np.minimum(np.abs(pred_front - front), inf_replace)
      diff_front_left = np.minimum(np.abs(pred_front_left - front_left), inf_replace)
      diff_front_right = np.minimum(np.abs(pred_front_right - front_right), inf_replace)
      diff_left = np.minimum(np.abs(pred_left - left), inf_replace)
      diff_right = np.minimum(np.abs(pred_right - right), inf_replace)

      return erfc(diff_front) + erfc(diff_front_left) + erfc(diff_front_right) + 2.5 * erfc(diff_left) + 2.5 * erfc(
        diff_right)

    new_weight = (5.0 + angle_erfc(0)) / 5.0
    self._weight = new_weight

    if not self.is_valid():
      self._weight *= 0.2

  def ray_trace(self, angle):
    """Returns the distance to the first obstacle from the particle."""

    def intersection_segment(x1, x2, y1, y2):
      point1 = np.array([x1, y1], dtype=np.float32)
      point2 = np.array([x2, y2], dtype=np.float32)
      v1 = self._pose[:2] - point1
      v2 = point2 - point1
      v3 = np.array([np.cos(angle + self._pose[YAW] + np.pi / 2.), np.sin(angle + self._pose[YAW] + np.pi / 2.)],
                    dtype=np.float32)
      t1 = np.cross(v2, v1) / np.dot(v2, v3)
      t2 = np.dot(v1, v3) / np.dot(v2, v3)
      if t1 >= 0. and t2 >= 0. and t2 <= 1.:
        return t1
      return float('inf')

    def intersection_cylinder(x, y, r):
      center = np.array([x, y], dtype=np.float32)
      v = np.array([np.cos(angle + self._pose[YAW] + np.pi), np.sin(angle + self._pose[YAW] + np.pi)],
                   dtype=np.float32)

      v1 = center - self._pose[:2]
      a = v.dot(v)
      b = 2. * v.dot(v1)
      c = v1.dot(v1) - r ** 2.
      q = b ** 2. - 4. * a * c
      if q < 0.:
        return float('inf')
      g = 1. / (2. * a)
      q = g * np.sqrt(q)
      b = -b * g
      d = min(b + q, b - q)
      if d >= 0.:
        return d
      return float('inf')

    d = min(intersection_segment(-WALL_OFFSET, -WALL_OFFSET, -WALL_OFFSET, WALL_OFFSET),
            intersection_segment(WALL_OFFSET, WALL_OFFSET, -WALL_OFFSET, WALL_OFFSET),
            intersection_segment(-WALL_OFFSET, WALL_OFFSET, -WALL_OFFSET, -WALL_OFFSET),
            intersection_segment(-WALL_OFFSET, WALL_OFFSET, WALL_OFFSET, WALL_OFFSET),
            intersection_cylinder(CYLINDER_POSITION[X], CYLINDER_POSITION[Y], CYLINDER_RADIUS))
    return d

  @property
  def pose(self):
    return self._pose

  @property
  def weight(self):
    return self._weight


class SimpleLaser(object):
  def __init__(self):
    rospy.Subscriber('/scan', LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 3.1  # 3.1 degrees cone of view (3 rays).
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
      return a <= x or x <= b;

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


class Motion(object):
  def __init__(self):
    self._previous_time = None
    self._delta_pose = np.array([0., 0., 0.], dtype=np.float32)
    rospy.Subscriber('/odom', Odometry, self.callback)

  def callback(self, msg):
    u = msg.twist.twist.linear.x
    w = msg.twist.twist.angular.z
    if self._previous_time is None:
      self._previous_time = msg.header.stamp
    current_time = msg.header.stamp
    dt = (current_time - self._previous_time).to_sec()
    self._delta_pose[X] += u * dt
    self._delta_pose[Y] += 0.
    self._delta_pose[YAW] += w * dt
    self._previous_time = current_time

  @property
  def ready(self):
    return True

  @property
  def delta_pose(self):
    ret = self._delta_pose.copy()
    self._delta_pose[:] = 0
    return ret


class GroundtruthPose(object):
  def __init__(self, name='turtlebot3_burger'):
    rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    self._name = name

  def callback(self, msg):
    idx = [i for i, n in enumerate(msg.name) if n == self._name]
    if not idx:
      raise ValueError('Specified name "{}" does not exist.'.format(self._name))
    idx = idx[0]
    self._pose[X] = msg.pose[idx].position.x
    self._pose[Y] = msg.pose[idx].position.y
    _, _, yaw = euler_from_quaternion([
      msg.pose[idx].orientation.x,
      msg.pose[idx].orientation.y,
      msg.pose[idx].orientation.z,
      msg.pose[idx].orientation.w])
    self._pose[YAW] = yaw

  @property
  def ready(self):
    return not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose


def run(args):
  rospy.init_node('localization')

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  particle_publisher = rospy.Publisher('/particles', PointCloud, queue_size=1)
  laser = SimpleLaser()
  motion = Motion()
  # Keep track of groundtruth position for plotting purposes.
  groundtruth = GroundtruthPose()
  pose_history = []
  with open('/tmp/gazebo_exercise.txt', 'w'):
    pass

  num_particles = 50
  particles = [Particle() for _ in range(num_particles)]

  frame_id = 0
  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not laser.ready or not motion.ready or not groundtruth.ready:
      rate_limiter.sleep()
      continue

    u, w = braitenberg(*laser.measurements)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    # Update particle positions and weights.
    total_weight = 0.
    delta_pose = motion.delta_pose
    for i, p in enumerate(particles):
      p.move(delta_pose)
      p.compute_weight(*laser.measurements)
      total_weight += p.weight

    # Low variance re-sampling of particles.
    new_particles = []
    random_weight = np.random.rand() * total_weight / num_particles
    current_boundary = particles[0].weight
    j = 0
    for m in range(len(particles)):
      next_boundary = random_weight + m * total_weight / num_particles
      while next_boundary > current_boundary:
        j = j + 1;
        if j >= num_particles:
          j = num_particles - 1
        current_boundary = current_boundary + particles[j].weight
      new_particles.append(copy.deepcopy(particles[j]))
    particles = new_particles

    # Publish particles.
    particle_msg = PointCloud()
    particle_msg.header.seq = frame_id
    particle_msg.header.stamp = rospy.Time.now()
    particle_msg.header.frame_id = '/odom'
    intensity_channel = ChannelFloat32()
    intensity_channel.name = 'intensity'
    particle_msg.channels.append(intensity_channel)
    for p in particles:
      pt = Point32()
      pt.x = p.pose[X]
      pt.y = p.pose[Y]
      pt.z = .05
      particle_msg.points.append(pt)
      intensity_channel.values.append(p.weight)
    particle_publisher.publish(particle_msg)

    # Log groundtruth and estimated positions in /tmp/gazebo_exercise.txt
    poses = np.array([p.pose for p in particles], dtype=np.float32)
    median_pose = np.median(poses, axis=0)
    pose_history.append(np.concatenate([groundtruth.pose, median_pose], axis=0))
    if len(pose_history) % 10:
      with open('/tmp/gazebo_exercise.txt', 'a') as fp:
        fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
        pose_history = []
    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs a particle filter')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
