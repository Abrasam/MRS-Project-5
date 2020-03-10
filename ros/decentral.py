#!/usr/bin/env python2

import argparse
import numpy as np
import rospy

import obstacle_avoidance
import divide_areas
import region_trade
import rrt
import full_coverage

# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


all_robots = []

use_locpose = False

class Robot:

  def __init__(self, id, name, original_grid, occupancy_grid, scaling, owned):
    self.id = id
    self.name = name
    self.original_grid = original_grid
    self.occupancy_grid = occupancy_grid
    self.scaling = scaling
    self.owned = owned
    self.publisher = rospy.Publisher('/' + name + '/cmd_vel', Twist, queue_size=5)
    self.laser = obstacle_avoidance.SimpleLaser(name=name)
    self.groundtruth = obstacle_avoidance.GroundtruthPose(name=name)
    self.path = []
    self.route_poses = None
    self.route_target_index = 0
    self.route_arrived = False
    self.locpose = full_coverage.LocalisationPose(name)

    self.meeting_times = {}

    pass

  # Use this instead of groundtruth as this will be replaced with Sam's thing
  def pose(self):
    if use_locpose:
      return self.locpose.pose
    else:
      return self.groundtruth.pose

  # Numpy array of position
  def position_array(self):
    pose = self.pose()
    return np.array([pose[0], pose[1]], dtype=np.float32)

  def check_ready(self):
    return self.laser.ready and self.groundtruth.ready and (not use_locpose or self.locpose.ready)

  # Tuple of grid index
  def scaled_grid_index(self):
    pose = self.pose()

    bigPos = self.original_grid.get_index(self.position_array())

    return int(np.round(bigPos[0] / self.scaling)), int(np.round(bigPos[1] / self.scaling))

  def send_move_command(self, u, w):
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    self.publisher.publish(vel_msg)

  def create_region_poses(self):
    sgi = self.scaled_grid_index()
    if self.owned[sgi] != 1:
      # Luke's code requires us to be inside the region
      return False

    new_edges = divide_areas.calculate_mst(self.occupancy_grid, self.owned.astype(np.int32), sgi)

    poses = divide_areas.generate_route_poses(
      new_edges, sgi, self.occupancy_grid, [sgi], self.owned, lines_plot=new_edges)
    scaled_poses = []
    for a, b, c in poses:
      a, b = self.original_grid.get_position(a * self.scaling, b * self.scaling)
      # Flip C in diagonal and vertical to match with Gazebo.
      c += np.pi / 2
      if c > np.pi:
        c -= 2 * np.pi
      scaled_poses.append((a, b, c))

    if not scaled_poses:
      self.route_poses = None
      return False
    else:
      self.route_poses = scaled_poses
      return True

  def move_on_region_route(self, speed, epsilon):
    current_target = self.route_poses[self.route_target_index]
    current_position = self.pose().copy()
    # Check if at target.
    distance = ((current_target[0] - current_position[0]) ** 2
                + (current_target[1] - current_position[1]) ** 2) ** 0.5

    if distance < divide_areas.ROBOT_RADIUS or self.route_arrived:
      # Keep moving for a bit
      self.route_arrived = True
      if np.absolute((current_target[2]) - current_position[2]) < (0.03):  # Within 3 degrees
        # print("Next")
        self.route_arrived = False
        self.route_target_index += 1
        self.route_target_index %= len(self.route_poses)
        current_target = self.route_poses[self.route_target_index]
        # print(current_target)
        v = full_coverage.get_velocity(current_position, current_target, speed)
        # v = np.array([1, 0])
        u, w = full_coverage.feedback_linearized(current_position.copy(), v, epsilon=epsilon)
        # u=0.5
        # w=0
      else:
        # print("Rotating")
        # Rotate to correct orientation
        u = 0
        difference = ((current_target[2] % (2 * np.pi)) - (current_position[2] % (2 * np.pi))) % (2 * np.pi)

        if difference < np.pi:
          # Difference heading to 0
          w = max(0.75, difference)
        else:
          remaining = 2 * np.pi - difference
          w = -1 * max(0.75, remaining)
        # w = 0.2 if ((current_target[2]) - current_position[2]) > 0 and (current_target[2] - current_position[2]) < np.pi else -0.2
    else:
      # print("Moving")
      v = full_coverage.get_velocity(current_position, current_target, speed)
      # v = np.array([1, 0])
      u, w = full_coverage.feedback_linearized(current_position, v, epsilon=epsilon)
      # u = 0.5
      # w = 0

      self.send_move_command(u, w)

  def send_linearized_move(self, x, y, epsilon):
    theta = self.pose()[2]

    u = x * np.cos(theta) + y * np.sin(theta)
    w = (1 / epsilon) * (-x * np.sin(theta) + y * np.cos(theta))
    self.send_move_command(u, w)

  def send_linearized_move_avoiding(self, x, y, epsilon):
    theta = self.pose()[2]

    u = x * np.cos(theta) + y * np.sin(theta)
    w = (1 / epsilon) * (-x * np.sin(theta) + y * np.cos(theta))

    [front, front_left, front_right, left, right] = self.laser.measurements

    # Rule based obstacle avoider
    if front < 0.4:
      u *= 0.5
      if front_left < front_right:
        w += 0.1
      else:
        w -= 0.1

    if front < 0.2 or front_left < 0.2 or front_right < 0.2:
      # void the path
      u = -0.1
      w = -0.5
      self.path = []

    for other in all_robots:
      if other.id == self.id:
        continue

      dist = np.linalg.norm(self.position_array() - other.position_array())
      if dist > 0.4:
        continue

      u *= 0.5

      # other theta + np.pi, aka the other way
      other_theta = (other.pose()[2] + 2 * np.pi) % (np.pi * 2) - np.pi
      angle_diff = other_theta - theta
      angle_diff = (angle_diff + np.pi) % (np.pi * 2) - np.pi

      if angle_diff < 0:
        w -= 0.1
      else:
        w += 0.1

    self.send_move_command(u, w)

  def move_rule_based(self):
    u, w = obstacle_avoidance.rule_based(*self.laser.measurements)
    self.send_move_command(u, w)

  def update_navigation(self):

    if self.route_poses is not None:
      self.move_on_region_route(0.3, 0.1)
    elif len(self.path) > 2:
      self.move_on_path(0.3, 0.1)
    else:
      if not self.create_region_poses():
        self.target_random_in_region()
        self.move_rule_based()

  def move_on_path(self, speed, epsilon):
    dist = np.linalg.norm(self.position_array() - self.path[0])

    if dist < 0.06:
      del self.path[0]
      if len(self.path) == 0:
        return

    # MISSING: Return the velocity needed to follow the
    # path defined by path_points. Assume holonomicity of the
    # point located at position.

    closest_index = 0
    closest_dist = 10000

    for index in range(0, len(self.path)):
      point = self.path[index]
      dist = np.linalg.norm(point - self.position_array())
      if dist < closest_dist:
        closest_dist = dist
        closest_index = index

    target_index = closest_index + 1
    if target_index >= len(self.path):
      target_index = len(self.path) - 1

    target = self.path[target_index]

    # Keep emptying the points we've passed
    if target_index > 2:
      del self.path[0]

    diff = target - self.position_array()

    move = normalize(diff) * speed
    self.send_linearized_move_avoiding(move[0], move[1], epsilon)

  def rrt_target(self, target):
    start_node, final_node = rrt.rrt(self.pose(), target, self.original_grid)
    if final_node is not None:
      self.path = rrt.get_path(final_node)

    return final_node is not None

  def target_random_in_region(self):

    small_target = region_trade.random_owned_pos(self.owned)
    if small_target is None:
      return

    big_target = np.array([small_target[0] * self.scaling, small_target[1] * self.scaling], dtype=np.float32)
    target_pos = self.original_grid.get_position(big_target[0], big_target[1])

    self.rrt_target(target_pos)

  # Meet all the robots in the array, except if they have the same id
  def perform_meetings(self, robots, radio_range):
    us_pos = self.position_array()

    for other in robots:
      # don't meet ourself
      if other.id == self.id:
        continue

      other_pos = other.position_array()
      dist = np.linalg.norm(other_pos - us_pos)

      if dist < radio_range:
        self.meet(other)

  def meet(self, other):

    # Update meeting times, only meet at most once per second
    time = rospy.get_time()
    meeting_delay = 1  # delay between meetings in seconds

    if other.id in self.meeting_times and self.meeting_times[other.id] + meeting_delay > time:
      return

    self.meeting_times[other.id] = time

    # Only one robot handles the meeting
    if other.id < self.id:
      return

    print(self.name + " and " + other.name + " meet")

    us_owned = np.count_nonzero(self.owned)
    other_owned = np.count_nonzero(other.owned)

    both_owned = np.logical_and(self.owned, other.owned)

    if np.abs(us_owned - other_owned) < 10 and np.count_nonzero(both_owned) == 0:
      print(self.name + " and " + other.name + " own " + str(us_owned) + " and " + str(other_owned) + ", no trade")
      return
    else:
      print(self.name + " and " + other.name + " own " + str(us_owned) + " and " + str(other_owned))

    self.trade(other)
    print(self.name + " and " + other.name + " trade")

  def trade(self, other):
    combined = np.logical_or(self.owned, other.owned)

    pos_a = self.scaled_grid_index()
    pos_b = other.scaled_grid_index()

    if pos_a == pos_b:
      return

    try:
      us_owned, other_owned = region_trade.trade_regions(self.occupancy_grid.values, combined, pos_a, pos_b)

      self.owned = us_owned
      other.owned = other_owned
      self.route_poses = None
    except Exception as e:
      print("The trade failed")
      print(e.message)


def run(args):
  global all_robots

  rospy.init_node('decentral')

  rate_limiter = rospy.Rate(100)

  original_occupancy_grid, occupancy_grid, scaling = divide_areas.create_occupancy_grid(args)

  for_sale = np.zeros_like(occupancy_grid.values)
  for_sale[occupancy_grid.values == 0] = 1

  robots = []
  for i in range(0, 3):
    new_robot = Robot(i, "tb3_" + str(i), original_occupancy_grid, occupancy_grid, scaling, np.copy(for_sale))
    robots.append(new_robot)

  all_robots = robots

  while not rospy.is_shutdown():
    if any(not robot.check_ready() for robot in robots):
      rate_limiter.sleep()
      continue

    for robot in robots:
      # robot.move_rule_based()
      robot.update_navigation()
      robot.perform_meetings(robots, 2.0)

    rate_limiter.sleep()

  # TODO class for each robot, and then update each separately

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Decentralized simulation of robots')
  parser.add_argument('--map', action='store', default='../ros/world_map',
                      help='Which map to use.')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
