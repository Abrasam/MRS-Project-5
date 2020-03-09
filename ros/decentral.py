#!/usr/bin/env python2

import argparse
import numpy as np
import rospy

import obstacle_avoidance
import divide_areas
import region_trade

# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion


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

    self.meeting_times = {}

    pass

  # Use this instead of groundtruth as this will be replaced with Sam's thing
  def pose(self):
    return self.groundtruth.pose

  # Numpy array of position
  def position_array(self):
    pose = self.pose()
    return np.array([pose[0], pose[1]], dtype=np.float32)

  def check_ready(self):
    return self.laser.ready and self.groundtruth.ready

  # Tuple of grid index
  def scaled_grid_index(self):
    pose = self.pose()

    (bigX, bigY) = self.original_grid.get_index(pose)

    return np.round(bigX / self.scaling), np.round(bigY / self.scaling)

  def move_rule_based(self):
    u, w = obstacle_avoidance.rule_based(*self.laser.measurements)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    self.publisher.publish(vel_msg)

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

    if np.abs(us_owned - other_owned) < 10:
      print(self.name + " and " + other.name + " own " + str(us_owned) + " and " + str(other_owned) + ", no trade")
      return

    self.trade(other)
    print(self.name + " and " + other.name + " trade")

  def trade(self, other):
    combined = np.logical_or(self.owned, other.owned)

    pos_a = self.scaled_grid_index()
    pos_b = other.scaled_grid_index()

    us_owned, other_owned = region_trade.trade_regions(self.occupancy_grid.values, combined, pos_a, pos_b)

    self.owned = us_owned
    other.owned = other_owned


def run(args):
  rospy.init_node('decentral')

  rate_limiter = rospy.Rate(100)

  original_occupancy_grid, occupancy_grid, scaling = divide_areas.create_occupancy_grid(args)

  for_sale = np.zeros_like(occupancy_grid.values)
  for_sale[occupancy_grid.values == 0] = 1

  robots = []
  for i in range(0, 3):
    new_robot = Robot(i, "tb3_" + str(i), original_occupancy_grid, occupancy_grid, scaling, np.copy(for_sale))
    robots.append(new_robot)


  while not rospy.is_shutdown():
    if any(not robot.check_ready() for robot in robots):
      rate_limiter.sleep()
      continue

    for robot in robots:
      robot.move_rule_based()

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
