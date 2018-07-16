import numpy as np
from os import path


# Move forward robot
def move_forward(th, delt, v, om, ):
    dx = v * delt * np.cos(th + om * delt / 2)
    dy = v * delt * np.sin(th + om * delt / 2)
    dth = delt * om
    return np.array([[dx], [dy], [dth]])


# Read odometry data
file_path = path.dirname(path.realpath(__file__))
data_path = file_path + '/dataset_refined' + '/robot1_Odo.txt'
odo_array = np.loadtxt(data_path, dtype=float)

# Read measurement data
measurement_path = file_path + '/dataset_refined' + '/robot1_Mesurement.txt'
measurement_array = np.loadtxt(measurement_path, dtype=float)

# read ground truth data
ground_truth = file_path + '/dataset_refined' + '/robot1_grnd_truth.txt'
ground_truth_array = np.loadtxt(ground_truth, dtype=float)

# dictionary of Barcodes and Landmarks
codeDict = {'5': 1, '7': 19, '9': 13, '14': 2, '16': 15, '18': 11, '23': 5, '25': 12, '27': 7, '32': 4, '36': 10,
            '41': 3, '45': 18, '54': 8, '61': 17, '63': 20, '70': 9, '72': 6, '81': 14, '90': 16}

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
stateCov = np.zeros((3 * numLM + 3, 3 * numLM + 3))
stateCov[0:3, 0:3] = 0.001

for i in range(3 * numLM + 3):
    stateCov[i, i] = 0.65

# Generate reshape matrix
fx = np.hstack((np.eye(3), np.zeros((3, 3 * numLM))))

# Initial pose
mu_init = np.array([3.4652, -3.2492, 2.5189]).reshape((1, 3))
mu_old = mu_init
mu_old = fx.T.dot(mu_old.T)

# Path of motion
# Set initial motions
start_idx = 600
# Initialize state mean
stateMean = np.array([[ground_truth_array[start_idx, 1]],
                      [ground_truth_array[start_idx, 2]],
                      ground_truth_array[start_idx, 3]])
stateMean = fx.T.dot(stateMean)
# Initial robot velocity and heading angle
vel = odo_array[start_idx, 1]
ang_vel = odo_array[start_idx, 2]
mIdx = 0
for i in range(start_idx, 800):  # odo_array.shape[0]):
    # current time instance
    t = ground_truth_array[start_idx, 0]
    # Calculate mu_bar
    theta = mu_old[2, 0]
    poseUpdate = move_forward(theta, dt, vel, ang_vel)
    mu_new = mu_old + fx.T.dot(poseUpdate)
    vel = odo_array[i, 1]
    ang_vel = odo_array[i, 2]
    print('mu_old: {}'.format(mu_old[0:3, 0]))
    print('Diff motion {}'.format(diffMotion))

    # Calculate movement jacobian
    gt = np.array([[0, 0, vel * dt * np.cos(theta + ang_vel * dt / 2)],
                   [0, 0, vel * dt * np.sin(theta + ang_vel * dt / 2)],
                   [0, 0, 0]])
    G_t = np.eye(3 * numLM + 3) + fx.T.dot(gt.dot(fx))

    # Calculate motion covariance in control space
    M_t = np.array([[(alphas[0] * np.abs(vel) + alphas[1] * np.abs(vel)) ** 2, 0],
                    [0, (alphas[2] * np.abs(vel) + alphas[3] * np.abs(vel)) ** 2]])

    # Calculate jacobian to transform motion covariance in state space
    V_t = np.array([[np.cos(theta + dt * ang_vel / 2), -0.5 * np.sin(theta + dt * ang_vel / 2)],
                    [np.sin(theta + dt * ang_vel / 2), 0.5 * np.cos(theta + dt * ang_vel / 2)],
                    [0, 1]])

    # Update state covariance
    R_t = V_t.dot(M_t.dot(V_t.T))
    stateCovBar = (G_t.dot(stateCov.dot(G_t.T)) + fx.T.dot(R_t.dot(fx)))

    # Add features which are newly visible
    z = np.zeros((3, 1))
    while (measurement_array[mIdx, 0] - t < 0.005) and (mIdx < measurement_array.shape[0]):
        barCode = measurement_array[mIdx, 1]
        landmarkId = 0

        if barCode in codeDict.keys():
            landmarkId = codeDict['barCode']
        else:
            print('key not found')

        if landmarkId > 5 and landmarkId < 21:
            rangeVal = measurement_array[mIdx, 2]
            bearingVal = measurement_array[mIdx, 3]
            if int(z[2, 0]) == 0:
                z = np.array([[rangeVal], [bearingVal], [landmarkId - 5]])
            else:
                newZ = z = np.array([[rangeVal], [bearingVal], [landmarkId - 5]])
                z = np.hstack((z, newZ))
        mIdx += 1

    # if features are observed
    # loop over all features and compute Kalman gain
    if z[2, 0] != 0:
        S = np.zeros((np.shape(z)[1], 3, 3))
        zHat = np.zeros(3, np.shape(z)[1])

        for k in range(np.shape(z)[1]):
            j = z[2, k]

            # if the landmark has never been seen before
            # add it to the state vector
            if stateMeanBar[3 + j] == 0:
                landmark_pos = np.array([[z[0, k] * np.cos(z[1, k] + stateMeanBar[2])],
                                         [z[0, k] * np.sin(z[1, k] + stateMeanBar[2]],
                                         [0]])



