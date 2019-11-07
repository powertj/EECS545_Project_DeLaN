'''

    Generates character reference trajectories in [q, q_dot, q_ddot]

'''
import numpy as np
from scipy.io import loadmat
import copy

def J(q, l1=0.5, l2=0.5):
    ''' gets jacobian fcn - l1 and l2 are link lengths '''
    return np.asarray([[-l1 * np.sin(q[0]) - l2 * np.sin(q[0] + q[1]), -l2 * np.sin(q[0] + q[1])],
                     [l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]), l2 * np.cos(q[0] + q[1])]])


def load_data():
    data = loadmat('../data/mixoutALL_shifted.mat')

    # Char labels N x 1 vec of labels 1-20
    char_labels = data['consts']['charlabels'][0][0].T

    # Key maps label to character via key[label-1]
    key = data['consts']['key'][0][0].T.squeeze()

    # Trajectories in list of N trajectories - each may be a different length so cannot put into one matrix
    trajectories = data['mixout'].T

    return trajectories, char_labels, key


def convert_trajectory(trajectory, sample_rate=200):
    '''
        converts trajectory from end effector v to q_dot
        -- right now assuming that we start at same position every time -- can change
    '''
    q_init = np.asarray([np.pi / 4.0, np.pi / 2.0])
    q = q_init.copy()
    q_dot = []
    for i in range(len(trajectory)):
        # q_dot
        v = trajectory[i, :2]
        q_d = np.linalg.solve(J(q), v)
        q += q_d / 200.0
        q_dot.append(q_d)

    q_dot = np.asarray(q_dot)
    q = q_init + np.cumsum(q_dot, axis=0) / sample_rate
    q_ddot = np.concatenate((np.zeros((1, 2)), np.diff(q_dot, axis=0))) * sample_rate

    return np.concatenate((q, q_dot, q_ddot), axis=1)


if __name__ =='__main__':

    trajectories, labels, key = load_data()

    joint_trajectories = []
    for trajectory in trajectories.flatten():
        trajectory_joint_space = convert_trajectory(trajectory.T)
        joint_trajectories.append(trajectory_joint_space)

    joint_data = dict()
    joint_data['trajectories'] = joint_trajectories
    joint_data['char_label'] = labels
    joint_data['char_keys'] = key

    np.savez('../data/trajectories_joint_space.npz', joint_data)

