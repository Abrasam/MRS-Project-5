import argparse
import numpy as np
import divide_areas
import matplotlib.pylab as plt

import region_trade

def pos_to_grid(pos):
  p_min = -1.8
  p_max = 1.8

  x = int((pos[0] - p_min) * (14 / (p_max - p_min)))
  y = int((pos[1] - p_min) * (14 / (p_max - p_min)))

  if x < 0:
    x = 0
  if y < 0:
    y = 0
  if x > 13:
    x = 13
  if y > 13:
    y = 13

  # x = 13 - x
  # y = 13 - y
  # x = 13 - x

  # print(str((x, y)))

  t = x
  x = y
  y = t

  return x, y

def run(args):

  original_occupancy_grid, occupancy_grid, scaling = divide_areas.create_occupancy_grid(args)

  reg = plt.imread("../or_t1.png")

  reg_id = np.zeros((np.shape(reg)[0], np.shape(reg)[1]), dtype=np.int)

  col_0 = np.array([0.0, 0.50196081, 1.0])
  col_1 = np.array([0.48627451, 1.0, 0.47450981])
  col_2 = np.array([0.49803922, 0.0, 0.0])

  for x in range(0, np.shape(reg)[0]):
    for y in range(0, np.shape(reg)[1]):
      col = reg[(x, y)]

      if np.linalg.norm(col - col_0) < 0.1:
        reg_id[(13 - x, y)] = 1
      elif np.linalg.norm(col - col_1) < 0.1:
        reg_id[(13 - x, y)] = 2
      elif np.linalg.norm(col - col_2) < 0.1:
        reg_id[(13 - x, y)] = 3

  # region_trade.draw_grid(reg_id)


  rob_data = []

  for id in range(0, 3):
    data = np.genfromtxt('/tmp/gazebo_robot_nav_tb3_' +
                         str(id) + '.txt', delimiter=',')

    rob_data.append(data)

    inside_count = 0
    max_time = 50000

    for i in range(0, min(np.shape(data)[0], max_time)):
      pos = data[i]

      x, y = pos_to_grid(pos)
      if reg_id[(x, y)] == (id + 1):
        inside_count += 1

    total = min(np.shape(data)[0], max_time)
    frac = float(inside_count) / float(total)
    print("For robot " + str(id) + " fraction was " + str(inside_count) + " / " + str(total) + " = " + str(frac))

  covered = np.zeros_like(reg_id)
  total_cells = np.count_nonzero(reg_id)

  x_data = []
  after_1000 = []

  for i in range(0, 160000):
    for id in range(0, 3):
      data = rob_data[id]

      pos = data[i]

      x, y = pos_to_grid(pos)

      if reg_id[(x, y)] != 0:
        covered[(x, y)] = 1

      if i % 1000 == 0:
        x_data.append(i / 100)
        after_1000.append(100 * float(np.count_nonzero(covered)) / float(total_cells))


  total_covered = np.count_nonzero(covered)
  c_frac = float(total_covered) / float(total_cells)

  print("Covered cells was " + str(total_covered) + " / " + str(total_cells) + " = " + str(c_frac))

  plt.plot(x_data, after_1000)
  plt.show()

  pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Decentralized simulation of robots')
  parser.add_argument('--map', action='store', default='../ros/world_map',
                      help='Which map to use.')
  args, unknown = parser.parse_known_args()
  run(args)