from divide_areas import create_occupancy_grid
import numpy as np

import matplotlib.pylab as plt


ROBOT_RADIUS = 0.105 / 2.

def plot_map_cover(filename):
    original_occupancy_grid, _, _ = create_occupancy_grid(0, mapname="small_map")
    cover_grid_total, _, _ = create_occupancy_grid(0, mapname="small_map")

    cover_grid_1, _, _ = create_occupancy_grid(0, mapname="small_map")
    cover_grid_2, _, _ = create_occupancy_grid(0, mapname="small_map")
    cover_grid_3, _, _ = create_occupancy_grid(0, mapname="small_map")

    total_to_cover = original_occupancy_grid.total_free()

    data1 = np.genfromtxt('../EvaluationPlots/%s/gazebo_robot_nav_tb3_0.txt' % filename, delimiter=',')
    data2 = np.genfromtxt('../EvaluationPlots/%s/gazebo_robot_nav_tb3_1.txt' % filename, delimiter=',')
    data3 = np.genfromtxt('../EvaluationPlots/%s/gazebo_robot_nav_tb3_2.txt' % filename, delimiter=',')

    covered = np.zeros((data1.shape[0], 4))# column for each robot and one for overall
    for index, d in enumerate([data1, data2, data3]):
        print(d.shape)
        for i in range(d.shape[0]):
            print(i)
            x, y = d[i,:2]
            res = original_occupancy_grid.resolution
            #print(res, 2.0*ROBOT_RADIUS/res)
            #print("xy", x,y)
            for a in np.linspace(x - ROBOT_RADIUS, x + ROBOT_RADIUS,10):
                for b in np.linspace(y - ROBOT_RADIUS, y + ROBOT_RADIUS, 10):
                    #print(a, b, ((a - x)**2 + (b - y)**2) < ROBOT_RADIUS**2)
                    if ((a - x)**2 + (b - y)**2) <= ROBOT_RADIUS**2:
                        #covered_locations.append((a, b))
                        if original_occupancy_grid.is_free((a, b)):
                            w, z = cover_grid_total.get_index((a, b))
                            cover_grid_total.values[w, z] = 3
                            [cover_grid_1, cover_grid_2, cover_grid_3][index].values[w, z] = 3
            covered[i, index] = np.sum([cover_grid_1, cover_grid_2, cover_grid_3][index].values == 3)

    covered[:, 3] = np.sum(covered, axis=1)
    covered /= total_to_cover
    covered *= 100
    print(covered)

    fig = plt.figure()

    # Data
    robots = ['tb3_0', 'tb3_1', 'tb3_2']
    colors = ['r', 'g', 'b', 'y']

    for i in range(3):
        plt.plot(data1[:, 3], covered[:, i], colors[i], label=robots[i])
        plt.legend()

    plt.plot(data1[:, 3], covered[:, 3], colors[3], label='overall')
    plt.legend()

    plt.xlabel('Time (s)')
    plt.ylabel('Map covered (%)')

    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([0,100])
    #plt.title("Progress of map coverage")
    plt.show()


if __name__ == "__main__":
    plot_map_cover("multi_u10_w25_4")
