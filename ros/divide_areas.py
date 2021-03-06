#!/usr/bin/env python

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
from scipy.optimize import fsolve
import math
import sys
from scipy.ndimage.measurements import label
from scipy import ndimage
import copy
#import rospy
import time
from copy import deepcopy



# Constants used for indexing.
X = 0
Y = 1
YAW = 2


# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
# GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
#START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
#MAX_ITERATIONS = 500

# Any orientation is good.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)


UP, DOWN, LEFT, RIGHT = [0, 1, 2, 3]


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


# Defines an occupancy grid.
class OccupancyGrid(object):
    def __init__(self, values, origin, resolution):
        self._original_values = values.copy()
        self._values = values.copy()
        # Inflate obstacles (using a convolution).
        inflated_grid = np.zeros_like(values)
        inflated_grid[values == OCCUPIED] = 1.
        w = 2 * int(ROBOT_RADIUS / resolution) + 1
        inflated_grid = scipy.signal.convolve2d(
            inflated_grid, np.ones((w, w)), mode='same')
        self._values[inflated_grid > 0.] = OCCUPIED  # TODO - add back
        self._origin = np.array(origin[:2], dtype=np.float32)
        self._origin -= resolution / 2.
        assert origin[YAW] == 0.
        self._resolution = resolution

    @property
    def values(self):
        return self._values

    @property
    def resolution(self):
        return self._resolution

    @property
    def origin(self):
        return self._origin

    def draw(self):
        plt.imshow(self._original_values.T, interpolation='none', origin='lower',
                   extent=[self._origin[X],
                           self._origin[X] +
                           self._values.shape[0] * self._resolution,
                           self._origin[Y],
                           self._origin[Y] + self._values.shape[1] * self._resolution])
        plt.set_cmap('gray_r')

    def get_index(self, position):
        idx = ((position - self._origin) / self._resolution).astype(np.int32)
        if len(idx.shape) == 2:
            idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
            idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
            return (idx[:, 0], idx[:, 1])
        idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
        idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
        return tuple(idx)

    def get_position(self, i, j):
        return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

    def np_get_position(self, array):
        result = array.copy()
        result = result.astype(np.float32)
        return result * self._resolution + self._origin

    def is_occupied(self, position):
        return self._values[self.get_index(position)] == OCCUPIED

    def is_free(self, position):
        return self._values[self.get_index(position)] == FREE

    def is_occupied_by_index(self, i, j):
        return self._values[i, j] == OCCUPIED

    def is_free_by_index(self, i, j):
        return self._values[i, j] == FREE

    def total_free(self, number_robots=0):
        return np.sum(self._values == FREE) - number_robots


def sample_random_position(occupancy_grid):
    position = np.zeros(2, dtype=np.float32)
    dim = occupancy_grid.values.shape

    x = np.random.randint(0, dim[0])
    y = np.random.randint(0, dim[1])
    pos = occupancy_grid.get_position(x, y)
    while not occupancy_grid.is_free(pos):
        x = np.random.randint(0, dim[0])
        y = np.random.randint(0, dim[1])
        pos = occupancy_grid.get_position(x, y)

    return pos


def draw_world(occupancy_grid, robot_locations, assignments, lines_plot={}, poses=[], line_multiplier=1):
    fig, ax = plt.subplots()
    occupancy_grid.draw()

    colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
               (1, 1, 0), (1, 0, 1), (0, 1, 1),
               (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
               (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
               (0.25, 0, 0), (0, 0.25, 0), (0, 0, 0.25),
               (0.25, 0.25, 0), (0.25, 0, 0.25), (0, 0.25, 0.25)
               ]

    for (i, j), v in np.ndenumerate(assignments):
        pos = occupancy_grid.get_position(i, j)
        from_origin = pos - occupancy_grid.origin
        position = occupancy_grid.origin + line_multiplier*from_origin
        if occupancy_grid.is_free(position):
            if v == 0:
                continue
            rectangle = plt.Rectangle(
                position, occupancy_grid.resolution*line_multiplier, occupancy_grid.resolution*line_multiplier, fc=colours[v])
            plt.gca().add_patch(rectangle)
            #plt.show()


    for pose in poses:
        x, y, angle = pose
        x, y = occupancy_grid.get_position(x, y)
        if angle == 0:
            plt.arrow(x, y - occupancy_grid.resolution /
                      8, 0, occupancy_grid.resolution / 4)
        elif angle == np.pi / 2:
            plt.arrow(x + occupancy_grid.resolution / 8,
                      y, -occupancy_grid.resolution / 4, 0)
        elif angle == np.pi:
            plt.arrow(x, y + occupancy_grid.resolution /
                      8, 0, -occupancy_grid.resolution / 4)
        elif angle == -np.pi / 2:
            plt.arrow(x - occupancy_grid.resolution / 8,
                      y, occupancy_grid.resolution / 4, 0)
        else:
            print("Unable to plot")
            sys.exit()

    for robot in robot_locations:
        plot_position = occupancy_grid.get_position(robot[0], robot[1])
        plt.scatter(plot_position[0], plot_position[1],
                    s=10, marker='o', color='black', zorder=1000)
        plt.scatter(plot_position[0], plot_position[1],
                    s=10, marker='o', color='black', zorder=1000)

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# @profile


def divide_grid(occupancy_grid, robots):
    # Based on J Intell Robot Syst (2017) 86:663-680
    # Kapoutsis2017_Article_DARPDivideAreasAlgorithmForOpt.pdf

    # Create E matrices
    # Initially build this using Euclidean distance.


    Es = [np.array(occupancy_grid.values.shape) for i in robots]

    for i in range(len(robots)):
        robot = occupancy_grid.get_position(*robots[i])  # * unpacks a tuple

        x, y = np.indices(Es[i])
        indices = np.stack([x, y], axis=2)
        positions = occupancy_grid.np_get_position(indices)
        Es[i] = np.sqrt(np.sum((positions - robot)**2, axis=2))
        #Es[i] /= np.linalg.norm(Es[i])

    ms = np.ones((len(robots)))
    weight_to_change = 0

    shape = occupancy_grid.values.shape
    size = shape[0] * shape[1]

    not_equal_split = 1 if (occupancy_grid.total_free(
        len(robots)) % len(robots)) != 0 else 0

    down_thres = (size - not_equal_split * (len(robots) - 1)) / \
        (size * len(robots))
    upper_thres = (size + not_equal_split) / (size * len(robots))

    limit = 30000
    for iter in range(limit):
        if iter == limit-1:
            return False
        assignments = np.argmin(np.stack([i for i in Es], axis=2), axis=2) + 1
        assignments = np.where(occupancy_grid.values != FREE, 0, assignments)

        # Check finished

        f = occupancy_grid.total_free(len(robots)) / len(robots)

        # Recaluate the m values that are used to adjust E.
        # k is number of assigned free cells to each robot
        """k = np.bincount(assignments)
        print(k)"""
        k = np.array([np.sum(assignments == i)
                      for i in range(len(robots) + 1)])

        k = np.delete(k, 0)

        k -= 1
        if k[0] == 37 and k[1] == 38:
            x = 1
        plain_errors = k / float(occupancy_grid.total_free(len(robots)))
        #np.delete(k, 0)
        # np.where(plain_errors < down_thres, divFairError[r] = downThres - plainErrors[r])
        div_fair_error = np.zeros(plain_errors.shape)
        div_fair_error[plain_errors < down_thres] = down_thres - \
            plain_errors[plain_errors < down_thres]
        div_fair_error[plain_errors >= down_thres] = upper_thres - \
            plain_errors[plain_errors >= down_thres]

        total_neg_perc = np.sum(np.absolute(
            div_fair_error[div_fair_error < 0]))
        total_neg_plain_errors = np.sum(plain_errors[div_fair_error < 0])

        if total_neg_plain_errors != 0.0:
            ms = np.where(div_fair_error < 0, 1.0 + (plain_errors / total_neg_plain_errors) * (total_neg_perc / 2.0),
                          1.0 - (plain_errors / total_neg_plain_errors) * (total_neg_perc / 2.0))
        else:
            ms = np.ones(div_fair_error.shape)
        #unique = np.delete(unique, 0)
        #k = np.delete(k, 0)
        print(k, f)
        Cs = []
        multi_section = np.zeros(k.shape)
        for r in range(len(robots)):
            # count connected components
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            labeled, ncomponents = label(assignments == r + 1, structure)
            if ncomponents == 1:
                C = np.ones(assignments.shape)
            else:
                multi_section[r] = 1
                #draw_world(occupancy_grid, robot_locations, assignments)
                robot = robots[r]
                value_of_correct_component = labeled[robot[0], robot[1]]

                # cells in the corect component
                R = 1 - (labeled == value_of_correct_component)
                # plt.set_cmap('gray_r')
                # plt.show()
                dist_R = ndimage.distance_transform_edt(
                    R) * occupancy_grid.resolution
                dist_R = (dist_R - np.min(dist_R)) / \
                    (np.max(dist_R) - np.min(dist_R)) + 1
                # plt.set_cmap('gray_r')
                # plt.show()
                Q = 1 - ((labeled != 0) & (labeled !=
                                           value_of_correct_component))
                dist_Q = ndimage.distance_transform_edt(
                    Q) * occupancy_grid.resolution
                dist_Q = (dist_Q - np.min(dist_Q)) / \
                    (np.max(dist_Q) - np.min(dist_Q))
                C = dist_R - dist_Q
                Cmin = np.min(C)
                Cmax = np.max(C)
                C = (C - Cmin) * (0.02 / (Cmax - Cmin)) + 0.99
                #C = np.ones(assignments.shape)
            Cs.append(C)

        Cs = np.array(Cs)

        # Before altering E, check if completed
        if not (1 in multi_section) and np.max(k) - np.min(k) <= not_equal_split:
            break

        for i in range(len(robots)):
            Es[i] = Es[i] * ms[i] * Cs[i]  # * random
        #Es[weight_to_change] /= np.linalg.norm(Es[weight_to_change])
        # print()
        weight_to_change += 1
        weight_to_change %= len(robots)
        if iter % 1000 == 0:
            #draw_world(occupancy_grid, robot_locations, assignments)
            pass

    return assignments


def calculate_mst(occupancy_grid, assignments, robot_location):

    # get all the points that are in the region
    region = assignments[robot_location[0], robot_location[1]]
    nodes = []
    edges = {}
    for i in range(0, assignments.shape[0]):
        for j in range(0, assignments.shape[1]):
            if assignments[i, j] == region:
                nodes.append((i, j))
                edges[(i, j)] = [None, None, None, None]  # up down left right
    for node in nodes:
        i, j = node
        # Each node lists how many direction
        if i == 0 or assignments[i - 1, j] != region:
            edges[(i, j)][UP] = 0
        elif assignments[i - 1][j] == region and edges[(i - 1, j)][UP] != None:
            edges[(i, j)][UP] = edges[(i - 1, j)][UP] + 1

        if i == assignments.shape[0] - 1 or assignments[i + 1, j] != region:
            edges[(i, j)][DOWN] = 0
        elif assignments[i + 1][j] == region and edges[(i + 1, j)][DOWN] != None:
            edges[(i, j)][DOWN] = edges[(i + 1, j)][DOWN] + 1

        if j == 0 or assignments[i, j - 1] != region:
            edges[(i, j)][LEFT] = 0
        elif assignments[i][j - 1] == region and edges[(i, j - 1)][LEFT] != None:
            edges[(i, j)][LEFT] = edges[(i, j - 1)][LEFT] + 1

        if j == assignments.shape[1] - 1 or assignments[i, j + 1] != region:
            edges[(i, j)][RIGHT] = 0
        elif assignments[i][j + 1] == region and edges[(i, j + 1)][RIGHT] != None:
            edges[(i, j)][RIGHT] = edges[(i, j + 1)][RIGHT] + 1

    for node in nodes[::-1]:
        i, j = node
        # Each node lists how many direction
        if i == 0 or assignments[i - 1, j] != region:
            edges[(i, j)][UP] = 0
        elif assignments[i - 1][j] == region and edges[(i - 1, j)][UP] != None:
            edges[(i, j)][UP] = edges[(i - 1, j)][UP] + 1

        if i == assignments.shape[0] - 1 or assignments[i + 1, j] != region:
            edges[(i, j)][DOWN] = 0
        elif assignments[i + 1][j] == region and edges[(i + 1, j)][DOWN] != None:
            edges[(i, j)][DOWN] = edges[(i + 1, j)][DOWN] + 1

        if j == 0 or assignments[i, j - 1] != region:
            edges[(i, j)][LEFT] = 0
        elif assignments[i][j - 1] == region and edges[(i, j - 1)][LEFT] != None:
            edges[(i, j)][LEFT] = edges[(i, j - 1)][LEFT] + 1

        if j == assignments.shape[1] - 1 or assignments[i, j + 1] != region:
            edges[(i, j)][RIGHT] = 0
        elif assignments[i][j + 1] == region and edges[(i, j + 1)][RIGHT] != None:
            edges[(i, j)][RIGHT] = edges[(i, j + 1)][RIGHT] + 1

    left = np.zeros_like(assignments)
    right = np.zeros_like(assignments)
    up = np.zeros_like(assignments)
    down = np.zeros_like(assignments)

    max_val = 0
    max_node = None
    direction = None
    for node in nodes:
        left[node[0], node[1]] = edges[node][LEFT]
        right[node[0], node[1]] = edges[node][RIGHT]
        up[node[0], node[1]] = edges[node][UP]
        down[node[0], node[1]] = edges[node][DOWN]
        if any([i > max_val for i in edges[node]]):
            max_val = max(edges[node])
            max_node = node
            direction = edges[node].index(max_val)

    nodes_used = []
    edges_used = {}
    # From each node in the tree, determine which new direction gives the longest path.
    while len(nodes_used) < len(nodes):
        #root = max_node

        # moving_node = root # highest value in up,down,left,right

        all_directions = np.stack((up, down, left, right))
        # apply mask so that only select node already in nodes_used.
        if len(nodes_used) > 0:
            mask = np.zeros(assignments.shape)
            for a, b in nodes_used:
                mask[a, b] = 1
            mask = np.stack((mask, mask, mask, mask))
            all_directions = np.where(mask, all_directions, 0)
        max_node = np.unravel_index(
            all_directions.argmax(), all_directions.shape)
        direction = max_node[0]
        moving_node = max_node[1:]
        counter = all_directions[direction, moving_node[0], moving_node[1]]

        for i in range(counter + 1):

            if moving_node not in nodes_used:
                nodes_used.append(moving_node)
            if i != counter:
                if moving_node not in edges_used:
                    edges_used[moving_node] = []
                edges_used[moving_node].append(direction)
            # Adjust the values perpendicular to direction.
            if direction in [UP, DOWN]:
                # Set self up, down to 0
                if i == 0:
                    if direction == UP:
                        up[moving_node[0], moving_node[1]] = 0
                    if direction == DOWN:
                        down[moving_node[0], moving_node[1]] = 0
                else:
                    up[moving_node[0], moving_node[1]] = 0
                    down[moving_node[0], moving_node[1]] = 0
                if i == counter:
                    if direction == UP:
                        down[moving_node[0]-1, moving_node[1]] = 0
                    if direction == DOWN:
                        up[moving_node[0]+1, moving_node[1]] = 0
                left_node = (moving_node[0], moving_node[1] - 1)
                l1, l2 = left_node
                new_val = 0
                while l2 > 0 and assignments[l1, l2] == region:
                    right[l1, l2] = new_val
                    new_val += 1
                    l2 -= 1
                    if left_node in nodes_used:
                        break
                    left_node = (l1, l2)
                right_node = (moving_node[0], moving_node[1] + 1)
                r1, r2 = right_node
                new_val = 0
                while r2 < assignments.shape[1] - 1 and assignments[r1, r2] == region:
                    left[r1, r2] = new_val
                    new_val += 1
                    r2 += 1
                    if right_node in nodes_used:
                        break
                    right_node = (r1, r2)
            else:  # line is left to right

                if i == 0:
                    if direction == LEFT:
                        left[moving_node[0], moving_node[1]] = 0
                    if direction == RIGHT:
                        right[moving_node[0], moving_node[1]] = 0
                else:
                    left[moving_node[0], moving_node[1]] = 0
                    right[moving_node[0], moving_node[1]] = 0

                if i == counter:
                    if direction == LEFT:
                        right[moving_node[0], moving_node[1]-1] = 0
                    if direction == RIGHT:
                        left[moving_node[0], moving_node[1]+1] = 0

                up_node = (moving_node[0] - 1, moving_node[1])
                u1, u2 = up_node
                new_val = 0
                while u1 > 0 and assignments[u1, u2] == region:
                    down[u1, u2] = new_val
                    new_val += 1
                    u1 -= 1
                    if up_node in nodes_used:
                        break
                    up_node = (u1, u2)
                down_node = (moving_node[0] + 1, moving_node[1])
                d1, d2 = down_node
                new_val = 0
                while d1 < assignments.shape[0] - 1 and assignments[d1, d2] == region:
                    up[d1, d2] = new_val
                    new_val += 1
                    d1 += 1
                    if down_node in nodes_used:
                        break
                    down_node = (d1, d2)
            a, b = moving_node
            if direction == UP:
                moving_node = (a - 1, b)
            elif direction == DOWN:
                moving_node = (a + 1, b)
            elif direction == LEFT:
                moving_node = (a, b - 1)
            else:
                moving_node = (a, b + 1)
            # add reverse edge
            if i != counter:
                if moving_node not in edges_used:
                    edges_used[moving_node] = []
                edges_used[moving_node].append(
                    [DOWN, UP, RIGHT, LEFT][direction])
    return edges_used

def generate_route_poses(edges, robot_position, occupancy_grid, robot_positions, assignments, lines_plot=[]):
    poses = []
    start_edges = edges[robot_position]
    i, j = robot_position
    if UP in start_edges:
        # top left facing up
        pose = (i+0.25, j + 0.25, np.pi / 2)
        next_cell = (i - 1, j)
        entry_point = DOWN
    elif RIGHT in start_edges:
        # top right facing right
        pose = (i + 0.25, j + 0.75, 0)
        next_cell = (i, j + 1)
        entry_point = LEFT
    elif DOWN in start_edges:
        # bottom right facing down
        pose = (i + 0.75, j + 0.75, -np.pi / 2)
        next_cell = (i + 1, j)
        entry_point = UP
    elif LEFT in start_edges:
        # bottom left facing left
        pose = (i + 0.75, j+0.25, np.pi)
        next_cell = (i, j - 1)
        entry_point = RIGHT
    else:
        print("There has been a catastrophe.")
    poses.append(pose)
    start = True
    while poses[-1] != poses[0] or start:
        start = False
        cell_edges = edges[next_cell]
        i, j = next_cell
        # always move clockwise, so e.g. from DOWN means from down left.
        if entry_point == DOWN:
            if LEFT in cell_edges:
                poses.append((i + 0.75, j+0.25, np.pi))
                next_cell = (i, j - 1)
                entry_point = RIGHT
            else:
                # forward then another decision
                poses.append((i+0.75, j+0.25, np.pi/2))
                if UP in cell_edges:
                    poses.append((i + 0.25, j + 0.25, np.pi / 2))
                    next_cell = (i - 1, j)
                    entry_point = DOWN
                else:
                    # right then another decision
                    poses.append((i + 0.25, j + 0.25, 0))
                    if RIGHT in cell_edges:
                        poses.append((i + 0.25, j + 0.75, 0))
                        next_cell = (i, j + 1)
                        entry_point = LEFT
                    else:
                        # right, and check that all edges are here
                        poses.append((i + 0.25, j + 0.75, -np.pi / 2))
                        if DOWN in cell_edges:
                            poses.append((i + 0.75, j + 0.75, -np.pi / 2))
                            next_cell = (i + 1, j)
                            entry_point = UP
                        else:
                            print("The robot is in a cell with no edges.")
                            sys.exit()
        elif entry_point == LEFT:
            if UP in cell_edges:
                poses.append((i+0.25, j + 0.25, np.pi / 2))
                next_cell = (i - 1, j)
                entry_point = DOWN
            else:
                poses.append((i + 0.25, j + 0.25, 0))
                if RIGHT in cell_edges:
                    poses.append((i + 0.25, j + 0.75, 0))
                    next_cell = (i, j + 1)
                    entry_point = LEFT
                else:
                    poses.append((i + 0.25, j + 0.75, -np.pi / 2))
                    if DOWN in cell_edges:
                        poses.append((i + 0.75, j + 0.75, -np.pi / 2))
                        next_cell = (i + 1, j)
                        entry_point = UP
                    else:
                        poses.append((i + 0.75, j + 0.75, np.pi))
                        if LEFT in cell_edges:
                            poses.append((i + 0.75, j+0.25, np.pi))
                            next_cell = (i, j - 1)
                            entry_point = RIGHT
                        else:
                            print("The robot is in a cell with no edges.")
                            sys.exit()
        elif entry_point == UP:

            if RIGHT in cell_edges:
                poses.append((i + 0.25, j + 0.75, 0))
                next_cell = (i, j + 1)
                entry_point = LEFT
            else:
                poses.append((i + 0.25, j + 0.75, -np.pi / 2))
                if DOWN in cell_edges:
                    poses.append((i + 0.75, j + 0.75, -np.pi / 2))
                    next_cell = (i + 1, j)
                    entry_point = UP
                else:
                    poses.append((i + 0.75, j + 0.75, np.pi))
                    if LEFT in cell_edges:
                        poses.append((i + 0.75, j+0.25, np.pi))
                        next_cell = (i, j - 1)
                        entry_point = RIGHT
                    else:
                        poses.append((i + 0.75, j + 0.25, np.pi / 2))
                        if UP in cell_edges:
                            poses.append((i+0.25, j + 0.25, np.pi / 2))
                            next_cell = (i - 1, j)
                            entry_point = DOWN
                        else:
                            print("The robot is in a cell with no edges.")
                            sys.exit()
        elif entry_point == RIGHT:

            if DOWN in cell_edges:
                poses.append((i + 0.75, j + 0.75, -np.pi / 2))
                next_cell = (i + 1, j)
                entry_point = UP
            else:
                poses.append((i + 0.75, j + 0.75, np.pi))
                if LEFT in cell_edges:
                    poses.append((i + 0.75, j+0.25, np.pi))
                    next_cell = (i, j - 1)
                    entry_point = RIGHT
                else:
                    poses.append((i + 0.75, j + 0.25, np.pi / 2))
                    if UP in cell_edges:
                        poses.append((i+0.25, j + 0.25, np.pi / 2))
                        next_cell = (i - 1, j)
                        entry_point = DOWN
                    else:
                        poses.append((i + 0.25, j + 0.25, 0))
                        if RIGHT in cell_edges:
                            poses.append((i + 0.25, j + 0.75, 0))
                            next_cell = (i, j + 1)
                            entry_point = LEFT
                        else:

                            print("The robot is in a cell with no edges.")
                            sys.exit()
        else:
            print("Should not have reached here")
            sys.exit()
    # Remove any poses that have the same angle as the previous pose
    new_poses = []
    for i in range(len(poses)):
        if len(new_poses) == 0:
            new_poses.append(poses[i])
            continue
        if poses[i][2] == new_poses[-1][2]:
            continue
        else:
            new_poses.append(poses[i])
    if new_poses[-1] != new_poses[0]:
        new_poses.append(new_poses[0])

    return new_poses


def create_occupancy_grid(args, mapname = None):
    print(os.getcwd())
    print(os.listdir("."))
    if mapname == None:
        mapname = args.map
    with open(mapname + '.yaml') as fp:
        data = yaml.load(fp)
    img = read_pgm(os.path.join(os.path.dirname(mapname), data['image']))
    occupancy_grid = np.empty_like(img, dtype=np.int8)
    occupancy_grid[:] = UNKNOWN
    occupancy_grid[img < .1] = OCCUPIED
    occupancy_grid[img > .9] = FREE

    # Expand the walls so that the robots have more clearance
    for _ in range(2):
        expanded_occupancy_grid = occupancy_grid.copy()
        for i in range(occupancy_grid.shape[0]):
            for j in range(occupancy_grid.shape[1]):
                free = True
                for a in range(i-1, i+2):
                    for b in range(j-1, j+2):
                        if a < 0 or b < 0 or a > occupancy_grid.shape[0]-1 or b > occupancy_grid.shape[1]-1:
                            continue
                        if occupancy_grid[a, b] != FREE:
                            free = False
                expanded_occupancy_grid[i, j] = FREE if free else OCCUPIED
        occupancy_grid = expanded_occupancy_grid.copy()

    # Transpose (undo ROS processing).
    occupancy_grid = occupancy_grid.T
    # Invert Y-axis.
    occupancy_grid = occupancy_grid[:, ::-1]

    original_occupancy_grid = OccupancyGrid(
        occupancy_grid, data['origin'], data['resolution'])

    occupancy_grid = OccupancyGrid(
        occupancy_grid, data['origin'], data['resolution'])

    # Shrink occupancy grid - make the minimum cell size 2*robot_width so that the robot can go forwards and backwards along it.
    # Add
    square_edge_size = np.array([ROBOT_RADIUS * 4.5, ROBOT_RADIUS * 4.5])
    square_edge_size /= occupancy_grid.resolution
    adjusted_edge_size = (occupancy_grid.values.shape //
                          square_edge_size).astype(np.int)
    cells_to_merge = occupancy_grid.values.shape[0] / adjusted_edge_size[0]

    new_world = np.zeros(adjusted_edge_size)

    for i in range(adjusted_edge_size[0]):
        istart = int(np.floor(i * cells_to_merge))
        iend = int(np.ceil((i + 1) * cells_to_merge))
        for j in range(adjusted_edge_size[1]):
            # Extra clearance for safety
            jstart = int(np.floor(j * cells_to_merge))
            jend = int(np.ceil((j + 1) * cells_to_merge))
            if np.any(occupancy_grid.values[istart:iend, jstart:jend] == OCCUPIED):
                new_world[i, j] = OCCUPIED
            elif np.any(occupancy_grid.values[istart:iend, jstart:jend] == UNKNOWN):
                new_world[i, j] = UNKNOWN
            else:
                new_world[i, j] = FREE

    occupancy_grid = OccupancyGrid(
        new_world, data['origin'], cells_to_merge * data['resolution'])

    return original_occupancy_grid, occupancy_grid, cells_to_merge

# Returns a function that tells the robot where it must be at any given time.
def create_route(poses, robot_speed, occupancy_grid):

    for i in range(len(poses)):
        a, b, c = poses[i]
        a, b = occupancy_grid.get_position(a, b)
        print(a, b)
        poses[i] = (a, b, c)

    total_distance = 0
    for i in range(len(poses)-2):
        f = poses[i]
        t = poses[i+1]
        dist = ((f[0]-t[0])**2 + (f[1]-t[1])**2)**0.5
        if f[2] == t[2]:
            # Staight line segment
            total_distance += dist
        else:
            # quarter of a circle turn
            total_distance += (np.pi*dist / 4.0)

    time = total_distance / robot_speed
    time_between_each = time / (len(poses)-1)


    def position(t):
        t = t % time
        segment = int(t / time_between_each)
        #print(segment)
        start_pose = poses[segment]
        end_pose = poses[segment + 1]
        fraction = (t % time_between_each) / time_between_each
        if start_pose[2] == end_pose[2]:
            # straight line
            pose = (start_pose[0] + fraction*(end_pose[0]-start_pose[0]),
                    start_pose[1] + fraction*(end_pose[1]-start_pose[1]),
                    start_pose[2])
        else:
            angle = (end_pose[2] - start_pose[2]) * fraction
            angle = end_pose[2] + np.pi + fraction*(end_pose[2] - start_pose[2])
            radius = np.absolute(start_pose[0]-end_pose[0])

            y_change = end_pose[1] - start_pose[1] # Assume to be the radius
            x_change = end_pose[0] - start_pose[0]
            #angle = (end_pose[2] - start_pose[2]) * fraction

            if start_pose[2] == 0 or start_pose[2] == np.pi:
                centre = (end_pose[0], start_pose[1])
            else:
                centre = (start_pose[0], end_pose[1])


            pose = (centre[0] - radius*np.sin(angle),
                    centre[1] + radius*np.cos(angle),
                    start_pose[2] + fraction*(end_pose[2]-start_pose[2]))
        #print(start_pose, end_pose)
        #print(pose)
        #print(fraction)
        #print()
        return pose
    #print(poses)
    return position

def divide(args, robot_locations, robot_speed):


    original_occupancy_grid, occupancy_grid, scaling = create_occupancy_grid(args)
    #robot_locations = [(3, 7), (8, 8)]

    print(original_occupancy_grid.values.shape)
    print(robot_locations)
    robot_locations = [original_occupancy_grid.get_index(i) for i in robot_locations]
    #robot_locations = [(48, 51), (80, 51), (74, 95)]
    for r in range(len(robot_locations)):
        a, b = robot_locations[r]
        robot_locations[r] = (int(a/scaling), int(b/scaling))
        #If square is not free then move over to nearest adjacent free cell.
        if not occupancy_grid.is_free_by_index(*robot_locations[r]):
            i, j = robot_locations[r]
            to_check = [(i-1, j), (i+1, j), (i, j+1), (i, j-1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
            done=False
            for new in to_check:
                if new[0] >= 0 and new[0] < occupancy_grid.values.shape[0]:
                    if new[1] >= 0 and new[1] < occupancy_grid.values.shape[1]:
                        if occupancy_grid.is_free_by_index(*new):
                            robot_locations[r] = new
                            done=True
                            break
            if not done:
                # Abandon. Robots need to move around more.
                return False


    print(robot_locations)

    # divide the cells between the robots.
    assignments = divide_grid(occupancy_grid, robot_locations)
    if assignments is False:
        return False

    edges_used = {}
    robot_edges = []
    robot_paths = []  # list of poses for the robot.
    routes = []
    for robot_location in robot_locations:
        new_edges = calculate_mst(occupancy_grid, assignments, robot_location)
        robot_edges.append(new_edges)
        #draw_world(occupancy_grid, robot_locations, assignments, lines_plot = new_edges)
        poses = generate_route_poses(
            new_edges, robot_location, occupancy_grid, robot_locations, assignments, lines_plot=new_edges)
        scaled_poses = []
        for a, b, c in poses:
            a, b = original_occupancy_grid.get_position(a*scaling, b*scaling)
            # Flip C in diagonal and vertical to match with Gazebo.
            c += np.pi/2
            if c > np.pi:
                c -= 2*np.pi
            scaled_poses.append((a, b, c))
        robot_paths.append(scaled_poses)

        pose_func = create_route(deepcopy(scaled_poses), robot_speed, original_occupancy_grid)

        routes.append(pose_func)
        #sys.exit()
        for key in new_edges:
            edges_used[key] = new_edges[key]

    # Reconvert back to orignal world
    print(robot_locations)
    for i in range(len(robot_locations)):
        a, b = robot_locations[i]
        robot_locations[i] = (a*scaling, b*scaling)

    adjusted_edges_used = {}

    for key in edges_used:
        a, b = key
        adjusted_edges_used[(scaling*a+occupancy_grid.resolution, scaling*b+occupancy_grid.resolution)] = edges_used[key]

    draw_world(original_occupancy_grid, robot_locations, assignments, lines_plot=adjusted_edges_used, line_multiplier=scaling)
    return robot_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provides routes to each robot to give full coverage.')
    parser.add_argument('--map', action='store', default='../ros/world_map',
                        help='Which map to use.')
    args, unknown = parser.parse_known_args()
    """f = create_route([1, 2, 3, 4, 1], 10)
    for i in range(0, 10):
        print(f(i))"""
    #divide(args, [(1.936162, 3.2647955), (3.8319607, 1.4526322), (-0.82950312, -1.8517277)], 450)
    """try:
        run(args)
    except rospy.ROSInterruptException:
        pass"""
