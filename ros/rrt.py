from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml

import divide_areas


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
MAX_ITERATIONS = 500


def sample_random_position(occupancy_grid):
  position = np.zeros(2, dtype=np.float32)

  valid = False

  while not valid:
    (x_size, y_size) = np.shape(occupancy_grid.values)
    x_index = np.random.randint(x_size)
    y_index = np.random.randint(y_size)
    index = (x_index, y_index)

    if occupancy_grid.values[index] != FREE:
      continue

    position = occupancy_grid.get_position(x_index, y_index)
    position[X] += np.random.rand() * occupancy_grid.resolution
    position[Y] += np.random.rand() * occupancy_grid.resolution
    valid = True

  # MISSING: Sample a valid random position (do not sample the yaw).
  # The corresponding cell must be free in the occupancy grid.

  return position


def adjust_pose(node, final_position, occupancy_grid):
  start_position = node.pose[:2]

  final_pose = node.pose.copy()
  final_pose[:2] = final_position
  final_node = Node(final_pose)

  center, radius, angle = find_correct_circle(node, final_node)
  final_pose[YAW] = angle
  final_node = Node(final_pose)

  def circle_angle(pos):
    offset = np.subtract(pos, center)
    return np.arctan2(offset[Y], offset[X])

  start_angle = circle_angle(start_position)
  final_angle = circle_angle(final_position)

  def circle_pos(angle):
    dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    return center + radius * dir

  def range_split(a, b):
    pos_a = circle_pos(a)
    pos_b = circle_pos(b)
    index_a = occupancy_grid.get_index(pos_a)
    index_b = occupancy_grid.get_index(pos_b)

    if occupancy_grid.values[index_a] != FREE:
      return False
    if occupancy_grid.values[index_b] != FREE:
      return False

    mid_angle = a + ((b - a) / 2)
    pos_mid = circle_pos(mid_angle)
    index_mid = occupancy_grid.get_index(pos_mid)

    # if (index_a == index_b) or np.abs(b - a) < 0.0001:
    if index_a == index_mid or index_mid == index_b:
      return True

    first_half = range_split(a, mid_angle)
    if not first_half:
      return False

    return range_split(mid_angle, b)

  if range_split(start_angle, final_angle):
    return final_node
  else:
    return None




# Defines a node of the graph.
class Node(object):
  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._cost = 0.

  @property
  def pose(self):
    return self._pose

  def add_neighbor(self, node):
    self._neighbors.append(node)

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node):
    self._parent = node

  @property
  def neighbors(self):
    return self._neighbors

  @property
  def position(self):
    return self._pose[:2]

  @property
  def yaw(self):
    return self._pose[YAW]
  
  @property
  def direction(self):
    return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, c):
    self._cost = c


def rrt(start_pose, goal_position, occupancy_grid):
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_node = None
  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, final_node
  graph.append(start_node)
  for _ in range(MAX_ITERATIONS): 
    position = sample_random_position(occupancy_grid)
    # With a random chance, draw the goal position.
    if np.random.rand() < .05:
      position = goal_position
    # Find closest node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    potential_parent = sorted(((n, np.linalg.norm(position - n.position)) for n in graph), key=lambda x: x[1])
    # Pick a node at least some distance away but not too far.
    # We also verify that the angles are aligned (within pi / 4).
    u = None
    for n, d in potential_parent:
      if d > .2 and d < 1.5 and n.direction.dot(position - n.position) / d > 0.70710678118:
        u = n
        break
    else:
      continue
    v = adjust_pose(u, position, occupancy_grid)
    if v is None:
      continue
    u.add_neighbor(v)
    v.parent = u
    graph.append(v)
    if np.linalg.norm(v.position - goal_position) < .2:
      final_node = v
      break
  return start_node, final_node


def perpendicular(v):
  w = np.empty_like(v)
  w[X] = -v[Y]
  w[Y] = v[X]
  return w


def find_circle(node_a, node_b):
  db = perpendicular(node_b.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_b.position
  return center, np.abs(radius)


# The correct circle that doesn't falsely use b's direction
def find_correct_circle(node_a, node_b):
  da = perpendicular(node_a.direction)
  d_between = node_a.position - node_b.position
  b_perp = normalize(perpendicular(d_between))
  mid = (node_a.position + node_b.position) / 2.

  center = intersect_lines(node_a.position, da, mid, b_perp)

  radius = np.linalg.norm(node_a.position - center)

  final_dir = perpendicular(node_b.position - center)
  angle = np.arctan2(final_dir[Y], final_dir[X])

  # Account for the circle direction of rotation
  if (node_a.position[X] - center[X]) * (node_b.position[Y] - center[Y]) - (node_a.position[Y] - center[Y]) * (node_b.position[X] - center[X]) < 0:
    angle = np.remainder(angle + 2 * np.pi, 2 * np.pi) - np.pi

  return center, radius, angle


def intersect_lines(a_pos, a_dir, b_pos, b_dir):
  const = a_pos - b_pos
  lam = (b_dir[Y] * const[X] - b_dir[X] * const[Y]) / (b_dir[X] * a_dir[Y] - b_dir[Y] * a_dir[X])
  center = a_pos + lam * a_dir

  return center


def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


def get_path(final_node):
  # Construct path from RRT solution.
  if final_node is None:
    return []
  path_reversed = []
  path_reversed.append(final_node)
  while path_reversed[-1].parent is not None:
    path_reversed.append(path_reversed[-1].parent)
  path = list(reversed(path_reversed))
  # Put a point every 5 cm.
  distance = 0.05
  offset = 0.
  points_x = []
  points_y = []
  for u, v in zip(path, path[1:]):
    center, radius, angle = find_correct_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    clockwise = np.cross(u.direction, du).item() > 0.
    # Generate a point every 5cm apart.
    da = distance / radius
    offset_a = offset / radius
    if clockwise:
      da = -da
      offset_a = -offset_a
      if theta2 > theta1:
        theta2 -= 2. * np.pi
    else:
      if theta2 < theta1:
        theta2 += 2. * np.pi
    angles = np.arange(theta1 + offset_a, theta2, da)
    offset = distance - (theta2 - angles[-1]) * radius
    points_x.extend(center[X] + np.cos(angles) * radius)
    points_y.extend(center[Y] + np.sin(angles) * radius)
  return zip(points_x, points_y)


def draw_solution(start_node, final_node=None):
  ax = plt.gca()

  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    du = u.direction
    plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    dv = v.direction
    plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    if np.cross(u.direction, du).item() > 0.:
      theta1, theta2 = theta2, theta1
    ax.add_patch(patches.Arc(center, radius * 2., radius * 2.,
                             theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
                             color=color, lw=lw))

  points = []
  s = [(start_node, None)]  # (node, parent).
  while s:
    v, u = s.pop()
    if hasattr(v, 'visited'):
      continue
    v.visited = True
    # Draw path from u to v.
    if u is not None:
      draw_path(u, v)
    points.append(v.pose[:2])
    for w in v.neighbors:
      s.append((w, v))

  points = np.array(points)
  plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  if final_node is not None:
    plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      draw_path(v.parent, v, color='k', lw=2)
      v = v.parent


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='map', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(args.map + '.yaml') as fp:
    data = yaml.load(fp)
  img = divide_areas.read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = UNKNOWN
  occupancy_grid[img < .1] = OCCUPIED
  occupancy_grid[img > .9] = FREE
  # Transpose (undo ROS processing).
  occupancy_grid = occupancy_grid.T
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]
  occupancy_grid = divide_areas.OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

  # Run RRT.
  start_node, final_node = rrt(START_POSE, GOAL_POSITION, occupancy_grid)

  # Plot environment.
  fig, ax = plt.subplots()
  occupancy_grid.draw()
  plt.scatter(.3, .2, s=10, marker='o', color='green', zorder=1000)
  draw_solution(start_node, final_node)
  plt.scatter(START_POSE[0], START_POSE[1], s=10, marker='o', color='green', zorder=1000)
  plt.scatter(GOAL_POSITION[0], GOAL_POSITION[1], s=10, marker='o', color='red', zorder=1000)
  
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - 4., 4. + .5])
  plt.ylim([-.5 - 4., 4. + .5])
  plt.show()
  