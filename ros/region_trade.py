import argparse
import numpy as np
import matplotlib.pylab as plt

import divide_areas


class TradeState:

  def __init__(self, grid, for_sale):
    self.grid = grid
    self.for_sale = for_sale
    self.total_for_sale = np.count_nonzero(for_sale)
    self.sold = 0


class Trader:
  def __init__(self, grid):
    self.owned = np.zeros_like(grid)
    self.frontier_grid = np.zeros_like(grid)
    self.has_free = False
    self.total = 0
    self.frontier = []


# grid = occupancy grid
# for sale = 1 if a region is for sale
# rob_a = robot a's position
# rob_b = robot b's position
def trade_regions(grid, for_sale, rob_a_pos, rob_b_pos):
  state = TradeState(grid, for_sale)

  a = Trader(grid)
  b = Trader(grid)

  total_for_sale = np.count_nonzero(for_sale)
  sold = 0

  def add_to_frontier(pos, us, other):

    # check in bounds
    if pos[0] < 0 or pos[1] < 0 or pos[0] >= np.shape(state.grid)[0] or pos[1] >= np.shape(state.grid)[1]:
      return

    # check is valid in grid, and not already in frontier or owned
    if not state.for_sale[pos] or state.grid[pos] or us.frontier_grid[pos] or us.owned[pos]:
      return

    us.frontier.append(pos)
    us.frontier_grid[pos] = 1

    # See if we have added a space not owned by the other side
    if not other.owned[pos] and not us.has_free:
      us.has_free = True

  def add_neighbours_to_frontier(pos, us, other):
    add_to_frontier((pos[0] + 1, pos[1]), us, other)
    add_to_frontier((pos[0] - 1, pos[1]), us, other)
    add_to_frontier((pos[0], pos[1] + 1), us, other)
    add_to_frontier((pos[0], pos[1] - 1), us, other)

  def buy_position(pos, us, other):
    us.owned[pos] = 1
    us.frontier_grid[pos] = 0
    us.total += 1
    state.sold += 1

    if other.owned[pos]:
      other.owned[pos] = 0
      other.total -= 1
      state.sold -= 1

    add_neighbours_to_frontier(pos, us, other)

  def check_owns_with_bounds(pos, us):
    if pos[0] < 0 or pos[1] < 0 or pos[0] >= np.shape(state.grid)[0] or pos[1] >= np.shape(state.grid)[1]:
      return False

    return us.owned[pos]

  # Check if we can buy a position, basically if position in Frontier is still viable
  # Aka do we still own a surrounding point?
  def can_buy(pos, us):
    return check_owns_with_bounds((pos[0] + 1, pos[1]), us) or check_owns_with_bounds((pos[0], pos[1] + 1), us) or \
           check_owns_with_bounds((pos[0] - 1, pos[1]), us) or check_owns_with_bounds((pos[0], pos[1] - 1), us)

  # It works for our map, probably not a good general solution because of e.g. 1 wide tunnels
  def wont_cutoff_other(pos, us, other):
    if not other.owned[pos]:
      return True

    neighbours_owned = 0
    neighbours_owned += check_owns_with_bounds((pos[0] + 1, pos[1]), other)
    neighbours_owned += check_owns_with_bounds((pos[0] - 1, pos[1]), other)
    neighbours_owned += check_owns_with_bounds((pos[0], pos[1] + 1), other)
    neighbours_owned += check_owns_with_bounds((pos[0], pos[1] - 1), other)

    if neighbours_owned <= 1:
      return True

    # Basic checks only here, could be more advanced
    if check_owns_with_bounds((pos[0] + 1, pos[1]), other) and check_owns_with_bounds((pos[0], pos[1] + 1), other) and \
        not check_owns_with_bounds((pos[0] + 1, pos[1] + 1), other):
      return False

    if check_owns_with_bounds((pos[0] - 1, pos[1]), other) and check_owns_with_bounds((pos[0], pos[1] + 1), other) and \
        not check_owns_with_bounds((pos[0] - 1, pos[1] + 1), other):
      return False

    if check_owns_with_bounds((pos[0] + 1, pos[1]), other) and check_owns_with_bounds((pos[0], pos[1] - 1), other) and \
        not check_owns_with_bounds((pos[0] + 1, pos[1] - 1), other):
      return False

    if check_owns_with_bounds((pos[0] - 1, pos[1]), other) and check_owns_with_bounds((pos[0], pos[1] - 1), other) and \
        not check_owns_with_bounds((pos[0] - 1, pos[1] - 1), other):
      return False

    if check_owns_with_bounds((pos[0] - 1, pos[1]), other) and check_owns_with_bounds((pos[0] + 1, pos[1]), other) and \
        not check_owns_with_bounds((pos[0], pos[1] + 1), other) and not check_owns_with_bounds((pos[0], pos[1] - 1), other):
      return False

    if check_owns_with_bounds((pos[0], pos[1] - 1), other) and check_owns_with_bounds((pos[0], pos[1] + 1), other) and \
        not check_owns_with_bounds((pos[0] + 1, pos[1]), other) and not check_owns_with_bounds((pos[0] - 1, pos[1]), other):
      return False

    return True

  # Buy the start positions
  buy_position(rob_a_pos, a, b)
  buy_position(rob_b_pos, b, a)

  # images = 0
  # count = 0

  while state.sold < state.total_for_sale:
    # count += 1

    # if count % 500 == 0 and images < 3:
    #   images += 1
    #   draw_grid(a.frontier_grid)
    #   draw_grid(b.frontier_grid)
    #   draw_grid(a.owned)
    #   draw_grid(b.owned)

    # Do some trade
    if a.total <= b.total and len(a.frontier) > 0:
      current_us = a
      current_other = b
    elif len(b.frontier) > 0:
      current_us = b
      current_other = a
    elif len(a.frontier) > 0:
      current_us = a
      current_other = b
    else:
      draw_grid(state.for_sale)
      draw_grid(a.owned)
      draw_grid(b.owned)
      draw_grid(a.frontier_grid)
      draw_grid(b.frontier_grid)
      raise Exception("Nothing else to explore, but we didn't finish selling")

    if not a.has_free and not b.has_free:
      draw_grid(state.for_sale)
      draw_grid(a.owned)
      draw_grid(b.owned)
      draw_grid(a.frontier_grid)
      draw_grid(b.frontier_grid)
      raise Exception("Just squabbling between themselves and not buying anything new")

    def frontier_loop():
      # Breadth first = forwards
      for index in range(0, len(current_us.frontier)):
        current_pos = current_us.frontier[index]

        # Check that it's not owned by the other side
        if current_us.has_free and current_other.owned[current_pos]:
          continue

        # Check if no longer reachable
        if not can_buy(current_pos, current_us):
          del current_us.frontier[index]
          continue

        if not wont_cutoff_other(current_pos, current_us, current_other):
          continue

        # Buy this position
        del current_us.frontier[index]
        buy_position(current_pos, current_us, current_other)
        return True

      return False

    # Loop twice, because if it turns out we can't buy anything that isn't owned by the other,
    # we need to loop twice to buy those tiles
    if not frontier_loop():
      if current_us.has_free:
        current_us.has_free = False
        frontier_loop()
      # If it's more serious than this, will get caught as both frontiers being empty anyway

  return a.owned, b.owned


def random_owned_pos(owned):
  all_pos = []

  for ((x, y), val) in np.ndenumerate(owned):
    if val == 1:
      all_pos.append((x, y))

  return all_pos[np.random.randint(0, len(all_pos))]


def draw_grid(grid):
  fig = plt.figure()

  plt.imshow(grid, interpolation='none', origin='lower')
  # plt.matshow(grid)
  plt.colorbar()
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Provides routes to each robot to give full coverage.')
  parser.add_argument('--map', action='store', default='../ros/world_map',
                      help='Which map to use.')
  args, unknown = parser.parse_known_args()

  original_occupancy_grid, occupancy_grid, scaling = divide_areas.create_occupancy_grid(args)

  for_sale = np.zeros_like(occupancy_grid.values)
  for_sale[occupancy_grid.values == 0] = 1

  draw_grid(occupancy_grid.values)
  draw_grid(for_sale)

  rob_a_pos = (34, 21)
  rob_b_pos = (4, 6)
  rob_c_pos = (31, 6)

  a_owned = for_sale
  b_owned = for_sale
  c_owned = for_sale

  for i in range(0, 20):
    rnum = np.random.randint(0, 3)

    rob_a_pos = random_owned_pos(a_owned)
    rob_b_pos = random_owned_pos(b_owned)
    rob_c_pos = random_owned_pos(c_owned)

    if rnum == 0:
      print("Trade a b")
      a_owned, b_owned = trade_regions(occupancy_grid.values, np.logical_or(a_owned, b_owned), rob_a_pos, rob_b_pos)
    elif rnum == 1:
      print("Trade b c")
      b_owned, c_owned = trade_regions(occupancy_grid.values, np.logical_or(b_owned, c_owned), rob_b_pos, rob_c_pos)
    elif rnum == 2:
      print("Trade a c")
      a_owned, c_owned = trade_regions(occupancy_grid.values, np.logical_or(a_owned, c_owned), rob_a_pos, rob_c_pos)

    final_grid = np.zeros_like(for_sale)
    final_grid[a_owned == 1] += 1
    final_grid[b_owned == 1] += 2
    final_grid[c_owned == 1] += 4
    draw_grid(final_grid)

  draw_grid(a_owned)
  draw_grid(b_owned)
  draw_grid(c_owned)

  final_grid = np.zeros_like(for_sale)
  final_grid[a_owned == 1] += 1
  final_grid[b_owned == 1] += 2
  final_grid[c_owned == 1] += 4

  print("a owns " + str(np.count_nonzero(a_owned)))
  print("b owns " + str(np.count_nonzero(b_owned)))
  print("c owns " + str(np.count_nonzero(c_owned)))

  draw_grid(final_grid)

  # divide_areas.draw_world(occupancy_grid, [], [])

  pass
