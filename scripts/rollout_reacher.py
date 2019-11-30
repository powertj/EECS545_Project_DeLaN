from reacher_ff_network import get_ff_network
import gym
import gym_cenvs
import numpy as np
import matplotlib.pyplot as plt
import torch


data = np.load('../data/trajectories_joint_space_no_gravity.npz',allow_pickle=True)
test_character = 25
test_traj = data['trajectories'][test_character]
ref_torques = data['torques'][test_character]
Nt = test_traj.shape[0]

# plot figure
#plt.plot(test_traj[:,0], test_traj[:,1])
#plt.title('reference character trajectory')
#plt.show()

env = gym.make('Reacher-v0')
observation = env.reset()
ff_net = get_ff_network()
actions = ff_net(torch.tensor(test_traj, dtype=torch.float32).reshape(Nt,1,1,6)).detach().numpy()

fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(ref_torques[:,0], label='Calculated', color='b')
axs[0].plot(actions[:,0,0,0], label='Predicted', color='r')
axs[0].legend()
axs[0].set_ylabel(r'$\tau_1\,(N-m)$')
axs[1].plot(ref_torques[:,1], label='Calculated', color='b')
axs[1].plot(actions[:,0,0,1], label='Predicted', color='r')
axs[1].legend()
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel(r'$\tau_2\,(N-m)$')
fig.suptitle('Feed Forward Network')
plt.show()

env.unwrapped.set_state(test_traj[0,0:4])
for t in range(0, Nt):
    action = actions[t,0,0]
    #action = ref_torques[t,:]
    observation, reward, done, _ = env.step(action)
    env.render()


input('Press enter to continue:')
env.close()
