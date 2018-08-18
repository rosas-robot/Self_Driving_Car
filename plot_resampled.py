from matplotlib import pyplot as plt
import numpy as np

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
