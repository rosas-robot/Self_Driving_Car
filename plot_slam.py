#################################################################
#         Plotting the Ground Truth of Robot1 and Landmarks #####
#################################################################

from matplotlib import pyplot as plt

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
