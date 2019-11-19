import numpy as np
import matplotlib.pyplot as plt

data = np.load('../data/trajectories_joint_space.npz', allow_pickle=True)
# data = np.load('../data/trajectories_concat_joint_space.npz', allow_pickle=True)

trajectories = data['trajectories']
torques = data['torques']

for i in range(5):
    traj = np.random.randint(0,trajectories.shape[0])
    plt.plot(torques[traj][:,0],label=r'$\tau_1$')
    plt.plot(torques[traj][:,1],label=r'$\tau_2$')
    # plt.plot(trajectories[traj][:,0],label=r'$q_1$')
    # plt.plot(trajectories[traj][:,1],label=r'$q_2$')

    plt.legend()
    plt.show()
    plt.close()


