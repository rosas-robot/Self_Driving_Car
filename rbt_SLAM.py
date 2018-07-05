##################################################
#   Extended Kalman Filter based SLAM            #
#     Using refined dataset of UTIAS             #
##################################################

import numpy as np
from matplotlib import pyplot as plt


# Move forward robot
def move_forward(th, dt, v, om,):
    dx = v*dt*np.cos(th + om*dt/2)
    dy = v*dt*np.sin(th + om*dt/2)
    dth = dt*om

    return np.array([dx, dy, dth])


measurement_data = np.loadtxt('robot1_Mesurement.txt')
print(measurement_data)


