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


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([-3, -3], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
MAX_ITERATIONS = 400


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

  center, radius, angle, arc_length = find_correct_circle(node, final_node)
  final_pose[YAW] = angle
  final_node = Node(final_pose)
  final_node.cost = node.cost + arc_length

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

  @pose.setter
  def pose(self, pose):
    self._pose = pose.copy()

  def add_neighbor(self, node):
    self._neighbors.append(node)

  def remove_neighbor(self, node):
    self._neighbors.remove(node)

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


def rrt(start_pose, is_goal, random_in_goal, occupancy_grid):
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_node = None
  graph.append(start_node)
  for _ in range(MAX_ITERATIONS): 
    position = sample_random_position(occupancy_grid)
    # With a random chance, draw the goal position.
    if np.random.rand() < .25:
      position = random_in_goal()
    # Find closest node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    potential_parent = sorted(((n, n.cost + np.linalg.norm(position - n.position)) for n in graph), key=lambda x: x[1])
    # Pick a node at least some distance away but not too far.
    # We also verify that the angles are aligned (within pi / 4).
    u = None
    for n, d in potential_parent:
      true_d = d - n.cost
      if .2 < true_d < 1.5 and n.direction.dot(position - n.position) / true_d > 0.70710678118:
        u = n
        break
    else:
      continue
    v = adjust_pose(u, position, occupancy_grid)
    if v is None:
      continue
    u.add_neighbor(v)
    v.parent = u

    # Rewiring only nodes that have no children, to avoid YAW changes
    potential_children = sorted(((n, np.linalg.norm(v.position - n.position)) for n in graph), key=lambda x : x[1])
    for n, d in potential_children:
      if d > 1.0:
        break
      if .2 < d and v.direction.dot(n.position - v.position) / d > 0.70710678118 and v.cost + d < n.cost and len(n.neighbors) == 0:
        new_n = adjust_pose(v, n.position, occupancy_grid)
        if new_n is not None and new_n.cost < n.cost:
          n.pose = new_n.pose
          n.parent.remove_neighbor(n)
          n.parent = v
          v.add_neighbor(n)
          n.cost = new_n.cost

    graph.append(v)

    # if np.linalg.norm(v.position - goal_position) < .2:
    if is_goal(v.position):
      if final_node is None or v.cost < final_node.cost:
        final_node = v
      # break
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



# Finding b's angle at the end
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
    if node_a.yaw > angle:
      arc_length = radius * (node_a.yaw - angle)
    else:
      arc_length = radius * (2 * np.pi - (angle - node_a.yaw))
  else:
    if node_a.yaw > angle:
      arc_length = radius * (2 * np.pi - (node_a.yaw - angle))
    else:
      arc_length = radius * (angle - node_a.yaw)

  return center, radius, angle, arc_length


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


def read_pgm(filename, byteorder='>'):
  """Read PGM file."""
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  return img.astype(np.float32) / 255.


def draw_solution(start_node, final_node=None):
  ax = plt.gca()

  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    du = u.direction
    plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    dv = v.direction
    plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    center, radius, _, _ = find_correct_circle(u, v)
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


