from divide_areas import create_occupancy_grid
import numpy as np

import matplotlib.pylab as plt


ROBOT_RADIUS = 0.105 / 2.

def plot_map_cover(filename, number_robots):
    original_occupancy_grid, _, _ = create_occupancy_grid(0, mapname="small_map")
    cover_grid_total, _, _ = create_occupancy_grid(0, mapname="small_map")

    cover_grid = []
    for i in range(number_robots):
        cover_grid.append(create_occupancy_grid(0, mapname="small_map")[0])

    total_to_cover = original_occupancy_grid.total_free()
    data = []
    for i in range(number_robots):
        data.append(np.genfromtxt('../EvaluationPlots/%s/gazebo_robot_nav_tb3_%s.txt' % (filename, str(i)), delimiter=','))

    covered = np.zeros((data[0].shape[0], number_robots+1))# column for each robot and one for overall

    number_rows = data[0].shape[0]
    coverage = []
    times = []
    for row in range(data[0].shape[0]):
        print(row)
        for robot in range(number_robots):
            if row >= data[robot].shape[0]:
                continue
            d = data[robot][row, :2]
            x, y = d
            for a in np.linspace(x - ROBOT_RADIUS, x + ROBOT_RADIUS,10):
                for b in np.linspace(y - ROBOT_RADIUS, y + ROBOT_RADIUS, 10):
                    #print(a, b, ((a - x)**2 + (b - y)**2) < ROBOT_RADIUS**2)
                    if ((a - x)**2 + (b - y)**2) <= ROBOT_RADIUS**2:
                        #covered_locations.append((a, b))
                        w, z = cover_grid_total.get_index((a, b))
                        cover_grid_total.values[w, z] = 3
        coverage.append(np.sum(cover_grid_total.values == 3))
        times.append(data[0][row, -1])


    for i in range(len(coverage)):
        coverage[i] = 100.0 * coverage[i]  / float(total_to_cover)
    times = np.array(times)
    coverage = np.array(coverage)

    plt.plot(times, coverage, 'r')

    plt.xlabel('Time (s)')
    plt.ylabel('Map covered (%)')

    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([0,100])
    #plt.title("Progress of map coverage")
    plt.show()


if __name__ == "__main__":
    plot_map_cover("single_u10_w25_4", 3)
