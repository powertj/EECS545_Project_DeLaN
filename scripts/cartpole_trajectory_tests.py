import cartpole_delan_network as cdn
import cartpole_ff_network as cffn
import torch
from scipy.io import loadmat
from tqdm import tqdm
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from trajectory_selection import random_train_test_trajectories
np.random.seed(0)

# Load the dataset and choose test parameters
print("Loading dataset...")
fname = '../cartpole_traj_gen/data/cartpole_all.mat'
data = loadmat(fname)
print("Done!")
num_epoch = 200
num_samples_per_traj = 1
seeds = np.arange(2)
train_traj_range = np.arange(1,4)
cdn_loss = np.zeros((len(train_traj_range),len(seeds)))
cffn_loss = np.zeros(cdn_loss.shape)

# device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
device = "cpu" # Configure device

criterion = nn.MSELoss() # Specify the loss layer

i = 0
for num_train_trajs in tqdm(train_traj_range):
    for seed in seeds:
        torch.manual_seed(seed)
        train_trajectories, train_labels, test_trajectories, test_labels  = random_train_test_trajectories(data, num_train_labels=num_train_trajs, num_samples_per_label=1)
        TRAJ_train = TrajectoryDataset(data, train_trajectories, train_labels)
        TRAJ_test = TrajectoryDataset(data, test_trajectories, test_labels)
        trainloader = DataLoader(TRAJ_train, batch_size=None)
        testloader = DataLoader(TRAJ_test, batch_size=None)

        # create model for cartpole delan network and specify hyperparameters
        cdn_model = cdn.CartPole_DeLaN_Network().to(device)
        cdn_optimizer = optim.Adam(cdn_model.parameters(), lr=5e-3, weight_decay=1e-3)
        cdn_scheduler = optim.lr_scheduler.StepLR(cdn_optimizer, step_size=40, gamma=0.5)

        # create model for cartpole ff network and specify hyperparameters
        cffn_model = cffn.CartPole_FF_Network().to(device)
        cffn_optimizer = optim.Adam(cffn_model.parameters(), lr=5e-3, weight_decay=1e-3)
        cffn_scheduler = optim.lr_scheduler.StepLR(cffn_optimizer, step_size=40, gamma=0.5)

        # train and evaluate cartpole delan network
        cdn.train(cdn_model, criterion, trainloader, device, cdn_optimizer, cdn_scheduler, num_epoch)
        cdn_loss[i,seed] = cdn.evaluate(cdn_model, criterion, testloader, device)

        # train and evaluate cartpole ff network
        cffn.train(cffn_model, criterion, trainloader, device, cffn_optimizer, cffn_scheduler, num_epoch)
        cffn_loss[i,seed] = cffn.evaluate(cffn_model, criterion, testloader, device)
    i += 1

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
plt.xlabel('Unique Trajectory Types')
plt.legend()
plt.title('CartPole DeLaN vs FF-NN Test Error')
plt.savefig('cartpole_delan_vs_ff_test_error.png')
# plt.show()
plt.close()

np.savetxt('cdn_loss.txt',cdn_loss)
np.savetxt('cffn_loss.txt',cffn_loss)



