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

    for index in range(len(data)):
        d = data[index]
        print(d.shape)
        last_centre = None
        res = original_occupancy_grid.resolution
        for i in range(d.shape[0]):
            print(i)
            x, y = d[i,:2]
            """if last_centre == None:
                last_centre = (x, y)
            elif (x - last_centre[0])**2 + (y - last_centre[1])**2  < (res / 4.0)**2:
                continue
            last_centre = (x, y)"""

            for a in np.linspace(x - ROBOT_RADIUS, x + ROBOT_RADIUS,10):
                for b in np.linspace(y - ROBOT_RADIUS, y + ROBOT_RADIUS, 10):
                    #print(a, b, ((a - x)**2 + (b - y)**2) < ROBOT_RADIUS**2)
                    if ((a - x)**2 + (b - y)**2) <= ROBOT_RADIUS**2:
                        #covered_locations.append((a, b))
                        if original_occupancy_grid.is_free((a, b)):
                            w, z = cover_grid_total.get_index((a, b))
                            cover_grid_total.values[w, z] = 3
                            cover_grid[index].values[w, z] = 3
            covered[i, index] = np.sum(cover_grid[index].values == 3)
            covered[i, number_robots] = np.sum(cover_grid_total.values == 3)

    #covered[:, 3] = np.sum(covered, axis=1)
    covered /= total_to_cover
    covered *= 100
    print(covered)

    fig = plt.figure()

    # Data
    robots = ['tb3_0', 'tb3_1', 'tb3_2', 'tb3_3', 'tb3_4']
    colors = ['r', 'g', 'b', 'y', 'c']

    for i in range(number_robots):
        plt.plot(data[i][:, number_robots], covered[:, i], colors[i], label=robots[i])
        plt.legend()

    plt.plot(data[0][:, number_robots], covered[:, number_robots], colors[number_robots], label='overall')
    plt.legend()

    plt.xlabel('Time (s)')
    plt.ylabel('Map covered (%)')

    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([0,100])
    #plt.title("Progress of map coverage")
    plt.show()


if __name__ == "__main__":
    plot_map_cover("gt_n2", 2)
