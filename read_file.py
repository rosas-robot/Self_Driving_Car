import numpy as np
from os import path


# Move forward robot
def move_forward(th, delt, v, om,):
    dx = v*delt*np.cos(th + om*delt/2)
    dy = v*delt*np.sin(th + om*delt/2)
    dth = delt*om
    return np.array([[dx], [dy], [dth]])


# Read odometry data
file_path = path.dirname(path.realpath(__file__))
data_path = file_path + '/dataset_refined' + '/robot1_Odo.txt'
odo_array = np.loadtxt(data_path, dtype=float)

# Read measurement data
measurement_path = file_path + '/dataset_refined' + '/robot1_Measurement.txt'
measurement_array = np.loadtxt(measurement_path, dtype=float)

# Noise parameters of motion Model and Measurement Model
# Parameters of motion model
alphas = [0.1, 0.01, 0.18, 0.08, 0.0, 0.0]

# Measurement model noise parameters
Q_t = np.array([[11.80, 0.0, 0.0],
                [0.0, 0.18, 0.0],
                [0.0, 0.0, 1.0]])

# Sample time
dt = 0.02

# Number of landmarks
numLM = 15

# Initialize state covariance matrix
stateCov = np.zeros((3*numLM + 3, 3*numLM + 3));
stateCov[0:3, 0:3] = 0.001

for i in range(3*numLM + 3):
    stateCov[i, i] = 0.65

# Generate reshape matrix
fx = np.hstack((np.eye(3), np.zeros((3, 3*numLM))))

# Initial pose
mu_init = np.array([3.4652, -3.2492, 2.5189]).reshape((1, 3))
mu_old = mu_init
mu_old = fx.T.dot(mu_old.T)

# Path of motion
# Set initial motions
start_idx = 600
vel = odo_array[start_idx, 1]
ang_vel = odo_array[start_idx, 2]
mIdx = 0
for i in range(start_idx, 800): # odo_array.shape[0]):
    # current time instance
    t =
    # Calculate mu_bar
    theta = mu_old[2, 0]
    diffMotion = move_forward(theta, dt, vel, ang_vel)
    mu_new = mu_old + fx.T.dot(diffMotion)
    vel = odo_array[i, 1]
    ang_vel = odo_array[i, 2]
    print('mu_old: {}'.format(mu_old[0:3, 0]))
    print('Diff motion {}'.format(diffMotion))

    # Calculate movement jacobian
    gt = np.array([[0, 0, vel*dt*np.cos(theta + ang_vel*dt/2)],
                   [0, 0, vel*dt*np.sin(theta + ang_vel*dt/2)],
                   [0, 0, 0]])
    G_t = np.eye(3*numLM + 3) + fx.T.dot(gt.dot(fx))

    # Calculate motion covariance in control space
    M_t = np.array([[(alphas[0]*np.abs(vel) + alphas[1]*np.abs(vel))**2, 0],
                    [0, (alphas[2]*np.abs(vel) + alphas[3]*np.abs(vel))**2]])

    # Calculate jacobian to transform motion covariance in state space
    V_t = np.array([[np.cos(theta + dt * ang_vel / 2), -0.5*np.sin(theta + dt * ang_vel / 2)],
                    [np.sin(theta + dt * ang_vel / 2), 0.5*np.cos(theta + dt * ang_vel / 2)],
                    [0, 1]])

    # Update state covariance
    R_t = V_t.dot(M_t.dot(V_t.T))
    stateCovBar = (G_t.dot(stateCov.dot(G_t.T)) + fx.T.dot(R_t.dot(fx)))

    # Add features which are newly visible
    z = np.zeros((3, 1))
    while (measurement_array[mIdx, 0] - t < 0.005) and (mIdx < measurement_array.shape[0]):
        landMarkCode = measurement_array[mIdx, 1]
        landmarkId = 0

    # Update mu_old
    mu_old = mu_new
    print('updated mu_old: {}'.format(mu_old[0:3, 0]))
    print('------------')

