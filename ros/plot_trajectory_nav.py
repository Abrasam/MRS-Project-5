from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
import Image


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


if __name__ == '__main__':

  fig = plt.figure()

  # Data
  robots = ['tb3_0', 'tb3_1', 'tb3_2']
  colors = ['r', 'g', 'b']
  estcolors = ['c', 'm', 'y']

  for id in range(0, 3):
    robot = robots[id]
    color = colors[id]
    color2 = estcolors[id]
    data = np.genfromtxt('/tmp/gazebo_robot_nav_' + robot + '.txt', delimiter=',')

    plt.plot(data[:, 0], data[:, 1], color, label=robot)
    if data.shape[1] == 6:
      plt.plot(data[:, 3], data[:, 4], color2, label=robot+' estimated')
    plt.legend()


  #img = plt.imread("small_map.png")
  img = read_pgm("small_map.pgm")
  plt.imshow(img, extent=[-2, 2, -2, 2])


  # Axes and stuff
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-6.5, 6.5])
  plt.ylim([-6.5, 6.5])

  for id in range(0, 3):
    robot = robots[id]
    color = colors[id]
    color2 = estcolors[id]
    data = np.genfromtxt('/tmp/gazebo_robot_nav_' + robot + '.txt', delimiter=',')
    if data.shape[1] == 6:
      plt.figure()
      error = np.linalg.norm(data[:, :2] - data[:, 3:5], axis=1)
      plt.plot(error, c='b', lw=2)
      plt.ylabel('Error [m]')
      plt.xlabel('Timestep')
      plt.title(robot)

  plt.show()
