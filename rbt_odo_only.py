##################################################
#    Robot path if only odometry measurement     #
#              is available                      #
##################################################

import numpy as np
from matplotlib import pyplot as plt


# Move forward robot
def move_forward(th, dt, v, om,):
    dx = v*dt*np.cos(th + om*dt/2)
    dy = v*dt*np.sin(th + om*dt/2)
    dth = dt*om

    return np.array([dx, dy, dth])


# Read odometry data from the file
odo_data = []
odo_array = []
for line in open('Robot1_Odometry.dat', 'r'):
    data = line.rstrip().split()
    for i in range(len(data)):
        odo_data.append(float(data[i]))
    odo_array.append(odo_data)
    odo_data = []
odo_array = np.array(odo_array)

mu_init = np.array([3.57323240, -3.33283870, 2.34080000]).reshape((1, 3))
print(mu_init.shape)

mu_old = mu_init
delt = 0.001
vel = odo_array[0, 1]
angvel = odo_array[0, 2]
for i in range(odo_array.shape[0]-1):
    theta = mu_old[0, 2]
    mu_new = mu_old + move_forward(theta, delt, vel, angvel)
    mu_old = mu_new
    delt = odo_array[i+1, 0] - odo_array[i, 0]
    vel = odo_array[i+1, 1]
    angvel = odo_array[i+1, 2]
    print(mu_old)
