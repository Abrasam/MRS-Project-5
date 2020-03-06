#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import numpy as np
import rospy
import time

from scipy.stats import norm, multivariate_normal
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
WALL_OFFSET = 4.
CYLINDER_POSITION = np.array([.3, .2], dtype=np.float32)
CYLINDER_RADIUS = .3 + ROBOT_RADIUS

NUM_ROBOTS = 3


def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


class Particle(object):
  """Represents a particle."""

  def __init__(self):
    self._pose = np.zeros(3, dtype=np.float32)
    self._weight = 1.
    while True:
      self._pose[X] = np.random.random()*8-4
      self._pose[Y] = np.random.random()*8-4
      self._pose[YAW] = np.random.random()*2*np.pi
      if self.is_valid():
        break

  #def is_valid(self):
  #  return -2 < self._pose[X] < 2 and -2 < self._pose[Y] < 2 and (self._pose[X]-0.3)**2 + (self._pose[Y]-0.2)**2 > 0.3**2

  def is_valid(self):
    # MISSING: Implement a function that returns True if the current particle
    # position is valid. You might need to use this function in __init__()
    # and compute_weight().

    if np.abs(self._pose[X]) > WALL_OFFSET - ROBOT_RADIUS:
      return False
    elif np.abs(self._pose[Y]) > WALL_OFFSET - ROBOT_RADIUS:
      return False

    pos = np.array([self._pose[X], self._pose[Y]], dtype=np.float32)

    def inside_cylinder(cx, cy, cr):
      cyl_offset = np.subtract(pos, np.array([cx, cy], dtype=np.float32))
      cyl_dist = np.linalg.norm(cyl_offset)

      return cyl_dist < cr

    def inside_aa_box(x1, y1, x2, y2):
      return x1 < self._pose[X] < x2 and y1 < self._pose[Y] < y2

    if inside_cylinder(0.3, 0.2, 0.3) or\
       inside_cylinder(2.5, 0.5, 0.7) or\
       inside_cylinder(1.5, 2.5, 0.5) or\
       inside_cylinder(-2.0, 3.0, 0.3):
      return False

    if inside_aa_box(-2.15, -2.15, 2.15, -2.0) or\
       inside_aa_box(-2.15, -3.15, -2.0, 1.15):
      return False

    return True

  def move(self, delta_pose):
    self._pose[YAW] += delta_pose[YAW] * np.random.normal(loc=1, scale=0.2)
    self._pose[YAW] %= 2*np.pi

    dx = delta_pose[X] * np.random.normal(loc=1, scale=0.2)

    self._pose[X] += dx*np.cos(self._pose[YAW])
    self._pose[Y] += dx*np.sin(self._pose[YAW])

    if np.random.random() < 0.02:
        self.__init__()

  def refine_weight(self, messages):
    sigma_r = 0.8 #80cm
    sigma_a = 1 #60deg
    xi = [[sigma_r**2, 0],[0, sigma_a**2]]

    Q = 1
    maxw = 1

    for dist,ang,particles in messages:
      w = 0
      n = 0
      phi = multivariate_normal(mean=[dist, ang], cov=xi)
      for p in particles:
        dir_vec = np.array([np.cos(p.pose[YAW]), np.sin(p.pose[YAW])])
        between_vec = self.pose[0:2] - p.pose[0:2]
        dist = np.linalg.norm(between_vec)
        between_vec = normalize(between_vec)
        dot = dir_vec.dot(between_vec)
        if dot > 1:
          dot = 1
          print(dot)
        if dot < -1:
          dot = -1
          print(dot)
        ang = np.arccos(dot)
        if dir_vec.dot(np.array([-between_vec[Y],between_vec[X]])) > 0:
          ang = -ang
        pdf = phi.pdf([dist, ang])
        w += pdf * p.weight
      w /= len(particles) # normalise
      Q *= w
      maxw = max(maxw, w)
    Q /= maxw**len(messages) # normalise
    self._weight = self.weight*Q


  def compute_weight(self, front, front_left, front_right, left, right):
    sigma = .8
    variance = sigma ** 2.
    cap = lambda x: min(5,x)
    dist = lambda x: cap(self.ray_trace(x))
    prob = 1
    prob *= norm.pdf(cap(front), dist(0), sigma)
    prob *= norm.pdf(cap(front_left), dist(np.pi/4), sigma)
    prob *= norm.pdf(cap(front_right), dist(-np.pi/4), sigma)
    prob *= norm.pdf(cap(left), dist(np.pi/2), sigma)
    prob *= norm.pdf(cap(right), dist(-np.pi/2), sigma)
    prob /=1 # normalise to be within range 0-1
    self._weight = prob if self.is_valid() else 0

  def ray_trace(self, angle):
    """Returns the distance to the first obstacle from the particle."""
    def intersection_segment(x1, x2, y1, y2):
      point1 = np.array([x1, y1], dtype=np.float32)
      point2 = np.array([x2, y2], dtype=np.float32)
      v1 = self._pose[:2] - point1
      v2 = point2 - point1
      v3 = np.array([np.cos(angle + self._pose[YAW] + np.pi / 2.), np.sin(angle + self._pose[YAW]  + np.pi / 2.)],
                    dtype=np.float32)
      t1 = np.cross(v2, v1) / np.dot(v2, v3)
      t2 = np.dot(v1, v3) / np.dot(v2, v3)
      if t1 >= 0. and t2 >= 0. and t2 <= 1.:
        return t1
      return float('inf')

    def intersection_aa_box(x1, y1, x2, y2):
      px = self._pose[X]
      py = self._pose[Y]
      dist = float('inf')
      x_vel = np.cos(angle + self._pose[YAW])
      y_vel = np.sin(angle + self._pose[YAW])

      if abs(x_vel) > 0.0001:
        if px < x1 and x_vel > 0:
          y_move = (y_vel * (x1 - px) / x_vel)
          if y1 <= py + y_move <= y2:
            dist = np.sqrt((px - x1) ** 2 + y_move ** 2)
        elif px > x2 and x_vel < 0:
          y_move = (y_vel * (x2 - px) / x_vel)
          if y1 <= py + y_move <= y2:
            dist = np.sqrt((px - x2) ** 2 + y_move ** 2)

      if abs(y_vel) > 0.0001:
        if py < y1 and y_vel > 0:
          x_move = (x_vel * (y1 - py) / y_vel)
          if x1 <= px + x_move <= x2:
            dist = min(dist, np.sqrt(x_move ** 2 + (py - y1) ** 2))
        elif py > y2 and y_vel < 0:
          x_move = (x_vel * (y2 - py) / y_vel)
          if x1 <= px + x_move <= x2:
            dist = min(dist, np.sqrt(x_move ** 2 + (py - y2) ** 2))

      return dist

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

    d = min(
      intersection_segment(-WALL_OFFSET, -WALL_OFFSET, -WALL_OFFSET, WALL_OFFSET),
      intersection_segment(WALL_OFFSET, WALL_OFFSET, -WALL_OFFSET, WALL_OFFSET),
      intersection_segment(-WALL_OFFSET, WALL_OFFSET, -WALL_OFFSET, -WALL_OFFSET),
      intersection_segment(-WALL_OFFSET, WALL_OFFSET, WALL_OFFSET, WALL_OFFSET),
      intersection_cylinder(0.3, 0.2, 0.3),
      intersection_cylinder(2.5, 0.5, 0.7),
      intersection_cylinder(1.5, 2.5, 0.5),
      intersection_cylinder(-2.0, 3.0, 0.3),
      intersection_aa_box(-2.15, -2.15, 2.15, -2.0),
      intersection_aa_box(-2.15, -3.15, -2.0, 1.15)
    )
    return d

  @property
  def pose(self):
    return self._pose

  @property
  def weight(self):
    return self._weight

  def copy(self, particle):
    self._weight = particle.weight
    self._pose = particle.pose
    return self

  def set(self, pose):
    self._pose = pose
    return self


class SimpleLaser(object):
  def __init__(self, name=""):
    rospy.Subscriber('/scan' if name == "" else "/"+name+"/scan", LaserScan, self.callback)
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
  def __init__(self, name=""):
    self._previous_time = None
    self._delta_pose = np.array([0., 0., 0.], dtype=np.float32)
    rospy.Subscriber('/odom' if name == "" else "/" + name + "/odom", Odometry, self.callback)

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

  def ray_trace(self, angle):
    """Returns the distance to the first obstacle from the robot."""
    def intersection_segment(x1, x2, y1, y2):
      point1 = np.array([x1, y1], dtype=np.float32)
      point2 = np.array([x2, y2], dtype=np.float32)
      v1 = self._pose[:2] - point1
      v2 = point2 - point1
      v3 = np.array([np.cos(angle + self._pose[YAW] + np.pi / 2.), np.sin(angle + self._pose[YAW]  + np.pi / 2.)],
                    dtype=np.float32)
      t1 = np.cross(v2, v1) / np.dot(v2, v3)
      t2 = np.dot(v1, v3) / np.dot(v2, v3)
      if t1 >= 0. and t2 >= 0. and t2 <= 1.:
        return t1
      return float('inf')

    def intersection_aa_box(x1, y1, x2, y2):
      px = self._pose[X]
      py = self._pose[Y]
      dist = float('inf')
      x_vel = np.cos(angle + self._pose[YAW])
      y_vel = np.sin(angle + self._pose[YAW])

      if abs(x_vel) > 0.0001:
        if px < x1 and x_vel > 0:
          y_move = (y_vel * (x1 - px) / x_vel)
          if y1 <= py + y_move <= y2:
            dist = np.sqrt((px - x1) ** 2 + y_move ** 2)
        elif px > x2 and x_vel < 0:
          y_move = (y_vel * (x2 - px) / x_vel)
          if y1 <= py + y_move <= y2:
            dist = np.sqrt((px - x2) ** 2 + y_move ** 2)

      if abs(y_vel) > 0.0001:
        if py < y1 and y_vel > 0:
          x_move = (x_vel * (y1 - py) / y_vel)
          if x1 <= px + x_move <= x2:
            dist = min(dist, np.sqrt(x_move ** 2 + (py - y1) ** 2))
        elif py > y2 and y_vel < 0:
          x_move = (x_vel * (y2 - py) / y_vel)
          if x1 <= px + x_move <= x2:
            dist = min(dist, np.sqrt(x_move ** 2 + (py - y2) ** 2))

      return dist

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

    d = min(
      intersection_segment(-WALL_OFFSET, -WALL_OFFSET, -WALL_OFFSET, WALL_OFFSET),
      intersection_segment(WALL_OFFSET, WALL_OFFSET, -WALL_OFFSET, WALL_OFFSET),
      intersection_segment(-WALL_OFFSET, WALL_OFFSET, -WALL_OFFSET, -WALL_OFFSET),
      intersection_segment(-WALL_OFFSET, WALL_OFFSET, WALL_OFFSET, WALL_OFFSET),
      intersection_cylinder(0.3, 0.2, 0.3),
      intersection_cylinder(2.5, 0.5, 0.7),
      intersection_cylinder(1.5, 2.5, 0.5),
      intersection_cylinder(-2.0, 3.0, 0.3),
      intersection_aa_box(-2.15, -2.15, 2.15, -2.0),
      intersection_aa_box(-2.15, -3.15, -2.0, 1.15)
    )
    return d

def run(args):
  rospy.init_node('localization')

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(10)
  particle_publisher = [rospy.Publisher('/particles'+str(i), PointCloud, queue_size=1) for i in range(NUM_ROBOTS)]
  laser = [SimpleLaser("tb3_"+str(i)) for i in range(NUM_ROBOTS)]
  motion = [Motion("tb3_"+str(i)) for i in range(NUM_ROBOTS)]
  # Keep track of groundtruth position for plotting purposes.
  groundtruth = [GroundtruthPose("tb3_"+str(i)) for i in range(NUM_ROBOTS)]
  pose_history = [[] for i in range(NUM_ROBOTS)]
  for i in range(NUM_ROBOTS):
    with open('/tmp/gazebo_robot_tb3_' + str(i) + '.txt', 'w'):
      pass

  num_particles = 200
  particles = [[Particle() for _ in range(num_particles)] for i in range(NUM_ROBOTS)]

  frame_id = 0
  while not rospy.is_shutdown():
    for i in range(NUM_ROBOTS):
      # Make sure all measurements are ready.
      if not laser[i].ready or not motion[i].ready or not groundtruth[i].ready:
        rate_limiter.sleep()
        continue

      messages = []
      for j in range(NUM_ROBOTS):
        if i != j:
          dir_vec = np.array([np.cos(groundtruth[j].pose[YAW]), np.sin(groundtruth[j].pose[YAW])])
          between_vec = groundtruth[i].pose[0:2] - groundtruth[j].pose[0:2]
          dist = np.linalg.norm(between_vec)
          between_vec = normalize(between_vec)
          dot = dir_vec.dot(between_vec)
          ang = np.arccos(dot)
          if dir_vec.dot(np.array([-between_vec[Y],between_vec[X]])) > 0:
            ang = -ang
          raytrace = groundtruth[j].ray_trace(ang)
          if raytrace >= dist:
            messages.append((dist*np.random.normal(loc=1, scale=0.1), ang*np.random.normal(loc=1, scale=0.1), list(map(lambda p: Particle().copy(p), particles[j])) if j == 0 else list(map(lambda p: p.set(groundtruth[j].pose+normalize(np.random.randn())), [Particle() for _ in range(num_particles)]))))

      # Update particle positions and weights.
      total_weight = 0.
      print(i)
      delta_pose = motion[i].delta_pose
      for _, p in enumerate(particles[i]):
        p.move(delta_pose)
        p.compute_weight(*laser[i].measurements)
        p.refine_weight(messages)
        total_weight += p.weight

      # Low variance re-sampling of particles.
      new_particles = []
      random_weight = np.random.rand() * total_weight / num_particles
      current_boundary = particles[i][0].weight
      j = 0
      for m in range(len(particles[i])):
        next_boundary = random_weight + m * total_weight / num_particles
        while next_boundary > current_boundary: 
          j = j + 1;
          if j >= num_particles:
            j = num_particles - 1
          current_boundary = current_boundary + particles[i][j].weight
        new_particles.append(copy.deepcopy(particles[i][j]))
      particles[i] = new_particles

      # Publish particles.
      particle_msg = PointCloud()
      particle_msg.header.seq = frame_id
      particle_msg.header.stamp = rospy.Time.now()
      particle_msg.header.frame_id = '/tb3_'+str(i)+'/odom'
      intensity_channel = ChannelFloat32()
      intensity_channel.name = 'intensity'
      particle_msg.channels.append(intensity_channel)
      for p in particles[i]:
        pt = Point32()
        pt.x = p.pose[X]
        pt.y = p.pose[Y]
        pt.z = .05
        particle_msg.points.append(pt)
        intensity_channel.values.append(p.weight)
      particle_publisher[i].publish(particle_msg)

      # Log groundtruth and estimated positions in /tmp/gazebo_exercise.txt
      poses = np.array([p.pose for p in particles[i]], dtype=np.float32)
      median_pose = np.median(poses, axis=0)
      pose_history[i].append(np.concatenate([groundtruth[i].pose, median_pose], axis=0))
      if len(pose_history[i]) % 10:
        with open('/tmp/gazebo_robot_tb3_' + str(i) + '.txt', 'a') as fp:
          fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history[i]) + '\n')
          pose_history[i] = []
      rate_limiter.sleep()
      frame_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs a particle filter')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
