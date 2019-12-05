import numpy as np
import matplotlib.pyplot as plt

train_traj_range = np.arange(1,4)
cdn_loss = np.loadtxt('cdn_loss.txt')
cffn_loss = np.loadtxt('cffn_loss.txt')

# statistics for cartpole delan
cdn_mean = np.mean(cdn_loss, axis=1)
cdn_sigma = np.std(cdn_loss, axis=1)
cdn_upper_95conf = cdn_mean + 2 * cdn_sigma
cdn_lower_95conf = np.maximum(cdn_mean - 2 * cdn_sigma, np.zeros(cdn_mean.shape))

# statistics for cartpole ff-nn
cffn_mean = np.mean(cffn_loss, axis=1)
cffn_sigma = np.std(cffn_loss, axis=1)
cffn_upper_95conf = cffn_mean + 2 * cffn_sigma
cffn_lower_95conf = np.maximum(cffn_mean - 2 * cffn_sigma, np.zeros(cffn_mean.shape))

# generate test error plot
plt.plot(train_traj_range, np.mean(cdn_loss, axis=1), c='red',label='CartPole DeLaN')
plt.plot(train_traj_range, np.mean(cffn_loss, axis=1), c='blue',label='CartPole FF-NN')
plt.fill_between(train_traj_range,cdn_lower_95conf,cdn_upper_95conf,where=cdn_upper_95conf >= cdn_lower_95conf, facecolor='red', interpolate=True, alpha=0.5)
plt.fill_between(train_traj_range,cffn_lower_95conf,cffn_upper_95conf,where=cffn_upper_95conf >= cffn_lower_95conf, facecolor='blue', interpolate=True, alpha=0.5)
plt.yscale('log')
plt.xticks(train_traj_range)
plt.ylabel('MSE')
plt.xlabel('Unique Training Trajectories')
plt.legend()
plt.title('CartPole DeLaN vs FF-NN Test Error')
plt.savefig('cartpole_delan_vs_ff_test_error_log.png')
plt.show()
plt.close()
