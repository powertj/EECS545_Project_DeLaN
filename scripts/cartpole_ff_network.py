import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm # Displays a progress bar
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader, random_split
import random
from dataset import TrajectoryDataset
# torch.manual_seed(0) # Fix random seed for reproducibility

def generate_train_test_indices(data, num_train_trajectories=1):
    indices = np.arange(data['trajectories'].shape[0])
    train_trajectories = np.random.choice(indices,size=num_train_trajectories,replace=False)
    test_trajectories = np.delete(indices, train_trajectories,axis=0)

    return list(train_trajectories), list(test_trajectories)

class CartPole_FF_Network(nn.Module):
    def __init__(self):
        super().__init__()
        h1_dim = 64
        h2_dim = 64

        self.fc1 = nn.Linear(6, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc_last = nn.Linear(h2_dim, 2)

    def forward(self,x):
        q = x
        x = F.relu(self.fc1(q))
        x = F.relu(self.fc2(x))
        x = self.fc_last(x)
        # The loss layer will be applied outside Network class
        return x

def train(model, loader, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label, _, _, _ in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    MSEs = []
    num_plots= 4
    i = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label, _, _, _ in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            MSE_error = criterion(pred, label)
            MSEs.append(MSE_error.item())
            if i < num_plots:
                fig, axs = plt.subplots(2, sharex=True)
                axs[0].plot(label[:,0],label='Calculated',color='b')
                axs[0].plot(pred[:,0],label='Predicted',color='r')
                axs[0].legend()
                axs[0].set_ylabel(r'$\tau_1\,(N-m)$')
                axs[1].plot(label[:,1],label='Calculated',color='b')
                axs[1].plot(pred[:,1],label='Predicted',color='r')
                axs[1].legend()
                axs[1].set_xlabel('Time Step')
                axs[1].set_ylabel(r'$\tau_2\,(N-m)$')
                fig.suptitle('CartPole Feed Forward Network')
                plt.show()
                plt.close()
                i += 1

    Ave_MSE = np.mean(np.array(MSEs))
    print("Average Evaluation MSE: {}".format(Ave_MSE))
    return Ave_MSE

if __name__ == '__main__':
    # Load the dataset and train and test splits
    print("Loading dataset...")
    fname = '../cartpole_traj_gen/data/cartpole_trajs_goal_1_to_2.mat'
    data = loadmat(fname)
    train_trajectories, test_trajectories = generate_train_test_indices(data, num_train_trajectories=4)
    TRAJ_train = TrajectoryDataset(data,train_trajectories)
    TRAJ_test = TrajectoryDataset(data,test_trajectories)
    print("Done!")
    trainloader = DataLoader(TRAJ_train, batch_size=None)
    testloader = DataLoader(TRAJ_test, batch_size=None)

    # create model and specify hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
    model = CartPole_FF_Network().to(device)
    criterion = nn.MSELoss() # Specify the loss layer
    # TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
    num_epoch = 200 # TODO: Choose an appropriate number of training epochs

    # train and evaluate network
    train(model, trainloader, num_epoch)
    evaluate(model, testloader)