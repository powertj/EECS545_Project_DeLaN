from cartpole_delan_network import get_model
import gym
import gym_cenvs
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import torch
import numpy as np
import code


data = loadmat('../cartpole_traj_gen/data/cartpole_trajs_goal_1_to_2.mat')
test_rollout = 10
test_traj = data['trajectories'][test_rollout]
ref_torques = data['torques'][test_rollout]
Nt = test_traj.shape[0]

# plot figure
#plt.plot(test_traj[:,0], test_traj[:,1])
#plt.title('reference character trajectory')
#plt.show()

env = gym.make('ContinuousCartpole-v0')
observation = env.reset()
delan_net = get_model()
pred_tau, pred_Hq_ddot, pred_c, pred_g = delan_net(torch.tensor(test_traj, dtype=torch.float32))
actions = pred_tau.detach().numpy()
savemat('../cartpole_traj_predicted.mat', {'actions': actions})

#fig, axs = plt.subplots(2, sharex=True)
#axs[0].plot(ref_torques[:,0], label='Calculated', color='b')
#axs[0].plot(actions[:,0], label='Predicted', color='r')
#axs[0].legend()
#axs[0].set_ylabel(r'$\tau_1\,(N-m)$')
#axs[1].plot(ref_torques[:,1], label='Calculated', color='b')
#axs[1].plot(actions[:,1], label='Predicted', color='r')
#axs[1].legend()
#axs[1].set_xlabel('Time Step')
#axs[1].set_ylabel(r'$\tau_2\,(N-m)$')
#fig.suptitle('Cartpole DeLan Network')
#plt.show()

env.reset()
obs = np.zeros((Nt,4))
for t in range(0, Nt):
    action = actions[t,0]
    #action = ref_torques[t,:]
    observation, reward, done, _ = env.step(action)
    obs[t,:] = observation
    env.render()

plt.plot(obs)
plt.legend(('x', 'theta', 'xdot', 'thetadot'))
plt.show()

env.close()