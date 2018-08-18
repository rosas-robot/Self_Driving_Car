#################################################################
#         Plotting the Ground Truth of Robot1 and Landmarks #####
#################################################################
from matplotlib import pyplot as plt
import numpy as np


def plot_ground_truth():
    # Read landmarks
    landmark_sig = []
    x_coord = []
    y_coord = []

    for line in open('Landmark_Groundtruth.dat', 'r'):
        data = line.rstrip().split()
        landmark_sig.append(float(data[0]))
        x_coord.append(float(data[1]))
        y_coord.append(float(data[2]))

    # Read Robot1 Poses
    rbt_tstamp = []
    rbt_x = []
    rbt_y = []
    rbt_th = []

    for line in open('Robot1_Groundtruth.dat', 'r'):
        data = line.rstrip().split()
        rbt_tstamp.append(float(data[0]))
        rbt_x.append(float(data[1]))
        rbt_y.append(float(data[2]))
        rbt_th.append(float(data[3]))

    # Plot landmarks and robot1
    plt.plot(x_coord, y_coord, 'b*')
    plt.plot(rbt_x, rbt_y, 'r-')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title('Ground truth of Landmark and Robot1')
    plt.grid()
    plt.savefig('ground_truth_plot.png')
    plt.show()


def plot_resampled():
    # Load data
    grndTrth = np.loadtxt('grnd_truth1.txt', dtype='float64')

    # Data array
    timeArray = []
    xArray = []
    yArray = []
    for i in range(grndTrth.shape[0]):
        timeArray.append(grndTrth[i, 0])
        xArray.append(grndTrth[i, 1])
        yArray.append(grndTrth[i, 2])

    # Plot data
    plt.plot(xArray, yArray, 'b')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Re-sampled data Robot1')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_ground_truth()
    plot_resampled()
