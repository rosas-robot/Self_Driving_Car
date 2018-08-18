import numpy as np
from os import path
import math


def conBear(oldBear):
    while oldBear < -np.pi:
        oldBear += 2*np.pi

    while oldBear > np.pi:
        oldBear -= 2*np.pi

    newBear = oldBear

    return newBear


# File path
file_path = path.dirname(path.realpath(__file__))

# Data file names and folders
DATASET_FOLDER = 'dataset_refined'
ODOMETRY_DATA = 'robot1_Odo.txt'
MEASUREMENT_DATA = 'robot1_Mesurement.txt'
GROUND_TRUTH = 'robot1_grnd_truth.txt'

# Read Odometry Data
data_path = file_path + '/' + DATASET_FOLDER + '/' + ODOMETRY_DATA
odo_array = np.loadtxt(data_path, dtype=float)

# Read Measurement Data
measurement_path = file_path + '/' + DATASET_FOLDER + '/' + MEASUREMENT_DATA
measurement_array = np.loadtxt(measurement_path, dtype=float)

# read ground truth data
ground_truth = file_path + '/' + DATASET_FOLDER + '/' + GROUND_TRUTH
ground_truth_array = np.loadtxt(ground_truth, dtype=float)

# dictionary of Barcodes and Landmarks
codeDict = {'5.0': 1, '14.0': 2, '41.0': 3, '32.0': 4, '23.0': 5, '72.0': 6, '27.0': 7, '54.0': 8, '70.0': 9, '36.0': 10, '18.0': 11, '25.0': 12,
            '9.0': 13, '81.0': 14, '16.0': 15, '90.0': 16, '61.0': 17, '45.0': 18, '7.0': 19, '63.0': 20}

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

# Initialize state estimation matrix
robotEstPose = np.zeros((ground_truth_array.shape[0], 4))

# Initialize state covariance matrix
stateCov = np.zeros((3 * numLM + 3, 3 * numLM + 3))
stateCov[0:3, 0:3] = 0.001

for i in range(3 * numLM + 3):
    stateCov[i, i] = 0.65

# Generate reshape matrix
fx = np.hstack((np.eye(3), np.zeros((3, 3 * numLM))))

# Path of motion
# Set initial motions
start_idx = 599

# Initialize state mean
stateMean = np.array([[ground_truth_array[start_idx, 1]],
                      [ground_truth_array[start_idx, 2]],
                      [ground_truth_array[start_idx, 3]]])
stateMean = fx.T.dot(stateMean)

# Initial robot velocity and heading angle
mIdx = 0
for i in range(start_idx, ground_truth_array.shape[0]):  # odo_array.shape[0]):
    # current time instance
    t = ground_truth_array[i, 0]

    #  update movement vector
    ut = [odo_array[i, 1], odo_array[i, 2]]

    # update robot bearing
    theta = stateMean[2, 0]
    rot = dt * ut[1]
    halfRot = rot / 2
    trans = ut[0] * dt

    # poseUpdate = move_forward(theta, dt, trans, rot)
    poseUpdate = np.array([[trans*np.cos(theta + halfRot)],
                           [trans*np.sin(theta + halfRot)],
                           [rot]])
    stateMeanBar = stateMean + fx.T.dot(poseUpdate)

    # Calculate movement jacobian
    gt = np.array([[0, 0, -trans * np.sin(theta + halfRot)],
                   [0, 0, trans * np.cos(theta + halfRot)],
                   [0, 0, 0]])
    G_t = np.eye(3 * numLM + 3) + fx.T.dot(gt.dot(fx))

    # Calculate motion covariance in control space
    M_t = np.array([[(alphas[0] * np.abs(ut[0]) + alphas[1] * np.abs(ut[1])) ** 2, 0],
                    [0, (alphas[2] * np.abs(ut[0]) + alphas[3] * np.abs(ut[1])) ** 2]])

    # Calculate jacobian to transform motion covariance in state space
    V_t = np.array([[np.cos(theta + halfRot), -0.5 * np.sin(theta + halfRot)],
                    [np.sin(theta + halfRot), 0.5 * np.cos(theta + halfRot)],
                    [0, 1]])

    # Update state covariance
    R_t = V_t.dot(M_t.dot(V_t.T))
    stateCovBar = (G_t.dot(stateCov.dot(G_t.T)) + fx.T.dot(R_t.dot(fx)))

    # Add features which are newly visible
    z = np.zeros((3, 1))
    while (measurement_array[mIdx, 0] - t < 0.005) and (mIdx < measurement_array.shape[0]-1):
        print('mIdx: {}'.format(mIdx))
        barCode = measurement_array[mIdx, 1]
        landmarkId = 0

        if str(barCode) in codeDict.keys():
            landmarkId = codeDict[str(barCode)]
        else:
            print('key not found')

        if landmarkId > 5:
            if landmarkId < 21:
                rangeVal = measurement_array[mIdx, 2]
                bearingVal = measurement_array[mIdx, 3]
                if int(z[2, 0]) == 0:
                    z = np.array([[rangeVal], [bearingVal], [landmarkId - 5]])
                else:
                    newZ = np.array([[rangeVal], [bearingVal], [landmarkId - 5]])
                    z = np.hstack((z, newZ))
        mIdx += 1

    # if features are observed
    # loop over all features and compute Kalman gain
    if z[2, 0] > 0:    # do no know why Z[2, 0] > 1, in my sense it has to be 0
        # S = np.zeros((np.shape(z)[1], 3, 3))
        zHat = np.zeros((3, z.shape[1]))

        for k in range(z.shape[1]):
            j = int(z[2, k])

            # if the landmark has never been seen before
            # add it to the state vector
            if stateMeanBar[2 + j, 0] == 0.0:
                landmark_pos = np.array([[z[0, k] * np.cos(z[1, k] + stateMeanBar[2, 0])],
                                         [z[0, k] * np.sin(z[1, k] + stateMeanBar[2, 0])],
                                         [0]])
                stateMeanBar[3*j:3*j+3] = np.array([[stateMeanBar[0, 0] + landmark_pos[0, 0]],
                                                    [stateMeanBar[1, 0] + landmark_pos[1, 0]],
                                                    [0]])
            # compute predicted range and bearing
            delta = np.array([[stateMeanBar[3 * j, 0] - stateMeanBar[0, 0]],
                              [stateMeanBar[3 * j + 1, 0] - stateMeanBar[1, 0]]])
            q = np.asscalar(delta.T.dot(delta))

            # predicted range to landmark
            r = np.asscalar(np.sqrt(q))

            # predicted bearing to landmark
            predBear = conBear(math.atan2(delta[1, 0], delta[0, 0]) - stateMeanBar[2, 0])

            zHat[:, k] = np.array([[r], [predBear], [j]]).reshape((3, ))

            h_t = np.array([[-r*np.asscalar(delta[0, 0]), -r*delta[1, 0], 0, r*delta[0, 0], r*delta[1, 0], 0],
                           [delta[1, 0], -delta[0, 0], -q, -delta[1, 0], delta[0, 0], 0],
                           [0, 0, 0, 0, 0, q]])

            F1 = np.vstack((np.eye(3), np.zeros((3, 3), dtype=float)))
            F2 = np.vstack((np.zeros((3, 3)), np.eye(3)))
            Fxj = np.hstack((F1, np.zeros((6, 3*j-3)), F2, np.zeros((6, 3*numLM-3*j))))

            H_t = (1/q) * h_t.dot(Fxj)

            # Compute Kalman gain
            invTemp = H_t.dot(stateCovBar.dot(H_t.T)) + Q_t
            invTemp = np.linalg.inv(invTemp)
            K = stateCovBar.dot(H_t.T.dot(invTemp))

            # Incorporate new measurement into state mean and covariance
            stateMeanBar = stateMeanBar + K.dot((z[:, k] - zHat[:, k]).T).reshape(48, 1)
            tempMat = K.dot(H_t)
            tempMat = np.eye(tempMat.shape[0]) - tempMat
            stateCovBar = tempMat.dot(stateCovBar)

    stateMean = stateMeanBar
    stateCov = stateCovBar
    robotEstPose[i, :] = [t, stateMean[0, 0], stateMean[1, 0], stateMean[2, 0]]

    if i == 605:
        print('\nIteration: {}\n'.format(i))

print(stateMean[0:3, 0])
print(robotEstPose[robotEstPose.shape[0]-1, :])
