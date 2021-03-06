from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
import Image


def plot_trajectory(occupancy_grid=None, assignments=None):

    fig = plt.figure()

    # Data
    robots = ['tb3_0', 'tb3_1', 'tb3_2']
    colors = ['r', 'g', 'b']
    estcolors = ['c', 'm', 'y']

    for id in range(0, 3):
        robot = robots[id]
        color = colors[id]
        color2 = estcolors[id]
        data = np.genfromtxt('/tmp/gazebo_robot_nav_' +
                             robot + '.txt', delimiter=',')

        plt.plot(data[:, 0], data[:, 1], color, label=robot)
        if data.shape[1] == 6:
            plt.plot(data[:, 3], data[:, 4], color2,
                     label=robot + ' estimated')
        plt.legend()

    scale = 0.9


    reg = plt.imread("../or_t1.png")
    plt.imshow(reg, extent=[-2.0 * scale, 2.0 * scale, -2.0 * scale, 2.0 * scale])

    img = plt.imread("small_map.png")
    img_trans = np.zeros((np.shape(img)[0], np.shape(img)[1], 4), dtype=np.float32)

    for x in range(0, np.shape(img)[0]):
      for y in range(0, np.shape(img)[1]):
        col = img[(x, y)]
        if np.linalg.norm(col) > 0.5:
          img_trans[(x, y)] = np.zeros(4, dtype=np.float32)
        else:
          img_trans[(x, y)] = np.array([col[0], col[1], col[2], 1.0], dtype=np.float32)

    plt.imshow(img_trans, extent=[-2.15, 2.15, -2.15, 2.15])

    # Axes and stuff
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.xlim([-6.5, 6.5])
    #plt.ylim([-6.5, 6.5])

    for id in range(0, 3):
        robot = robots[id]
        color = colors[id]
        color2 = estcolors[id]
        data = np.genfromtxt('/tmp/gazebo_robot_nav_' +
                             robot + '.txt', delimiter=',')
        if data.shape[1] == 6:
            plt.figure()
            error = np.linalg.norm(data[:, :2] - data[:, 3:5], axis=1)
            plt.plot(error, c='b', lw=2)
            plt.ylabel('Error [m]')
            plt.xlabel('Timestep')
            plt.title(robot)

    plt.show()


if __name__ == '__main__':
    plot_trajectory()
