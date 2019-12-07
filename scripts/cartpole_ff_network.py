import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm # Displays a progress bar
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from trajectory_selection import random_train_test_trajectories, select_train_test_trajectories
# torch.manual_seed(0) # Fix random seed for reproducibility

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

def train(model, criterion, loader, device, optimizer, scheduler, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for state, tau, _, _, _, _ in tqdm(loader):
            state = state.to(device)
            tau = tau.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(state) # This will call Network.forward() that you implement
            loss = criterion(pred, tau) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step() # Update trainable weights

        scheduler.step()
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, criterion, loader, device, show_plots=False, num_plots=1): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    MSEs = []
    i = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for state, tau, _, _, _, label in tqdm(loader):
            state = state.to(device)
            tau = tau.to(device)
            pred = model(state)
            MSE_error = criterion(pred, tau)
            MSEs.append(MSE_error.item())
            # if label == 3:
            #     np.savetxt('cartpole_ff_3_traj.txt', np.concatenate((tau,pred),axis=1))
            if show_plots:
                if i < num_plots:
                    fig, axs = plt.subplots(2, sharex=True)
                    axs[0].plot(tau[:,0],label='Calculated',color='b')
                    axs[0].plot(pred[:,0],label='Predicted',color='r')
                    axs[0].legend()
                    axs[0].set_ylabel(r'$\tau_1\,(N-m)$')
                    axs[1].plot(tau[:,1],label='Calculated',color='b')
                    axs[1].plot(pred[:,1],label='Predicted',color='r')
                    axs[1].set_xlabel('Time Step')
                    axs[1].set_ylabel(r'$\tau_2\,(N-m)$')
                    fig.suptitle('CartPole FF NN Trajectory {}'.format(label))
                    plt.show()
                    plt.close()
                    i += 1

    Ave_MSE = np.mean(np.array(MSEs))
    print("Average Evaluation MSE: {}".format(Ave_MSE))
    return Ave_MSE

if __name__ == '__main__':
    # Load the dataset and train and test splits
    print("Loading dataset...")
    # fname = '../cartpole_traj_gen/data/cartpole_all.mat'
    fname = '../cartpole_traj_gen/data/cartpole_all_200hz.mat'

    num_trials = 10
    MSEs = np.zeros(num_trials)
    for i in range(num_trials):
        data = loadmat(fname)
        # train_trajectories, train_labels, test_trajectories, test_labels  = random_train_test_trajectories(data, num_train_labels=1, num_samples_per_label=5)
        train_trajectories, train_labels, test_trajectories, test_labels  = select_train_test_trajectories(data, train_label_types=[2,3,4], num_samples_per_label=5)

        TRAJ_train = TrajectoryDataset(data, train_trajectories, train_labels)
        TRAJ_test = TrajectoryDataset(data, test_trajectories, test_labels)
        print("Done!")
        trainloader = DataLoader(TRAJ_train, batch_size=None)
        testloader = DataLoader(TRAJ_test, batch_size=None)

        # create model and specify hyperparameters
        device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device

        model = CartPole_FF_Network().to(device)
        criterion = nn.MSELoss() # Specify the loss layer
        # Modify the line below, experiment with different optimizers and parameters (such as learning rate)
        optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        num_epoch = 200 # Choose an appropriate number of training epochs

        # train and evaluate network
        train(model, criterion, trainloader, device, optimizer, scheduler, num_epoch)
        MSEs[i] = evaluate(model, criterion, testloader, device, show_plots=False)

    print('MSEs',MSEs)
    print('Mean MSE =',np.mean(MSEs))
    print('Std = ',np.std(MSEs))