from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
import Image


if __name__ == '__main__':

  fig = plt.figure()

  # Data
  robots = ['tb3_0', 'tb3_1', 'tb3_2']
  colors = ['r', 'g', 'b']

  for id in range(0, 3):
    robot = robots[id]
    color = colors[id]
    data = np.genfromtxt('/tmp/gazebo_robot_' + robot + '.txt', delimiter=',')

    plt.plot(data[:, 0], data[:, 1], color, label=robot)


  # Image
  # im = Image.open('./world_map.png')
  # im_height = im.size[1]
  #
  # im_arr = np.array(im).astype(np.float) / 255
  # fig.figimage(im_arr, 0, fig.bbox.ymax - im_height)

  img = plt.imread("world_map.png")
  plt.imshow(img, extent=[-4, 4, -4, 4])


  # Axes and stuff
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-5, 5])
  plt.ylim([-5, 5])

  plt.show()