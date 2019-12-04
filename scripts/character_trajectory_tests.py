import reacher_delan_network as rdn
import reacher_ff_network as rffn
import torch
from tqdm import tqdm
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from dataset import TrajectoryDataset
from torch.utils.data import DataLoader
np.random.seed(0)

# Load the dataset and choose test parameters
print("Loading dataset...")
data = np.load('../data/trajectories_joint_space.npz', allow_pickle=True)
print("Done!")
num_epoch = 150
num_samples_per_char = 1
seeds = np.arange(10)
train_chars_range = np.concatenate((np.array([1]),np.arange(2,19,step=2)))
rdn_loss = np.zeros((len(train_chars_range),len(seeds)))
rffn_loss = np.zeros(rdn_loss.shape)

# device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
device = "cpu" # Configure device

criterion = nn.MSELoss() # Specify the loss layer

i = 0
for num_train_chars in tqdm(train_chars_range):
    for seed in seeds:
        torch.manual_seed(seed)
        train_trajectories, test_trajectories = rdn.generate_train_test_indices(data, num_train_chars, num_samples_per_char)
        TRAJ_train = TrajectoryDataset(data,train_trajectories)
        TRAJ_test = TrajectoryDataset(data,test_trajectories)
        trainloader = DataLoader(TRAJ_train, batch_size=None)
        testloader = DataLoader(TRAJ_test, batch_size=None)

        # create model for reacher delan network and specify hyperparameters
        rdn_model = rdn.Reacher_DeLaN_Network().to(device)
        rdn_optimizer = optim.Adam(rdn_model.parameters(), lr=5e-3, weight_decay=1e-3)
        rdn_scheduler = optim.lr_scheduler.StepLR(rdn_optimizer, step_size=40, gamma=0.5)

        # create model for reacher ff network and specify hyperparameters
        rffn_model = rffn.Reacher_FF_Network().to(device)
        rffn_optimizer = optim.Adam(rffn_model.parameters(), lr=5e-2, weight_decay=1e-3)
        rffn_scheduler = optim.lr_scheduler.StepLR(rffn_optimizer, step_size=40, gamma=0.5)

        # train and evaluate reacher delan network
        rdn.train(rdn_model, criterion, trainloader, device, rdn_optimizer, rdn_scheduler, num_epoch)
        rdn_loss[i,seed] = rdn.evaluate(rdn_model, criterion, testloader, device)

        # train and evaluate reacher ff network
        rffn.train(rffn_model, criterion, trainloader, device, rffn_optimizer, rffn_scheduler, num_epoch)
        rffn_loss[i,seed] = rffn.evaluate(rffn_model, criterion, testloader, device)
    i += 1

# statistics for reacher delan
rdn_mean = np.mean(rdn_loss, axis=1)
rdn_sigma = np.std(rdn_loss, axis=1)
rdn_upper_95conf = rdn_mean + 2 * rdn_sigma
rdn_lower_95conf = rdn_mean - 2 * rdn_sigma

# statistics for reacher ff-nn
rffn_mean = np.mean(rffn_loss, axis=1)
rffn_sigma = np.std(rffn_loss, axis=1)
rffn_upper_95conf = rffn_mean + 2 * rffn_sigma
rffn_lower_95conf = rffn_mean - 2 * rffn_sigma

# generate test error plot
plt.plot(train_chars_range, np.mean(rdn_loss, axis=1), c='red',label='Reacher DeLaN')
plt.plot(train_chars_range, np.mean(rffn_loss, axis=1), c='blue',label='Reacher FF-NN')
plt.fill_between(train_chars_range,rdn_lower_95conf,rdn_upper_95conf,where=rdn_upper_95conf >= rdn_lower_95conf, facecolor='red', interpolate=True, alpha=0.5)
plt.fill_between(train_chars_range,rffn_lower_95conf,rffn_upper_95conf,where=rffn_upper_95conf >= rffn_lower_95conf, facecolor='blue', interpolate=True, alpha=0.5)
# plt.yscale('log')
plt.xticks(train_chars_range)
plt.ylabel('MSE')
plt.xlabel('Unique Training Characters')
plt.legend()
plt.title('Reacher DeLaN vs FF-NN Test Error')
plt.savefig('delan_vs_ff_test_error.png')
# plt.show()
plt.close()

np.savetxt('rdn_loss.txt',rdn_loss)
np.savetxt('rffn_loss.txt',rffn_loss)



