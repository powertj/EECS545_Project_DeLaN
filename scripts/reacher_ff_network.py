import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader, random_split
import random
# torch.manual_seed(0) # Fix random seed for reproducibility

# Load the dataset and train, val, test splits
print("Loading datasets...")

data = np.load('../data/trajectories_joint_space.npz',allow_pickle=True)
char_count = {}
char_indices = {}
test_chars = []
train_trajectories = []
test_trajectories = []

for i, label in enumerate(data['labels']):
    idx = label[0]
    letter = data['keys'][idx-1][0]
    if letter in char_count:
        char_count[letter] += 1
        char_indices[letter].append(i)

    else:
        test_chars.append(letter)
        char_count[letter] = 1
        char_indices[letter] = [i]

num_train_chars = 1
train_chars = []

for i in range(num_train_chars):
    if len(test_chars) > 0:
        train_char_idx = random.randint(0,len(test_chars)-1)
        train_char = test_chars.pop(train_char_idx)
        train_chars.append(train_char)
        train_trajectories.append(random.choice(char_indices[train_char]))

for test_char in test_chars:
        test_trajectories.append(random.choice(char_indices[test_char]))

# create training dataset
for count, i in enumerate(train_trajectories):
    if count == 0:
        joint_trajectories_concat = torch.from_numpy(data['trajectories'][i]).float()
        tau_list_concat = torch.from_numpy(data['torques'][i]).float()
    else:
        joint_trajectories_concat = torch.cat((joint_trajectories_concat,torch.from_numpy(data['trajectories'][i]).float()),dim=0)
        tau_list_concat = torch.cat((tau_list_concat,torch.from_numpy(data['torques'][i]).float()),dim=0)

TRAJ_train = TensorDataset(joint_trajectories_concat,tau_list_concat)

# create validation dataset
# for count, i in enumerate(val_characters):
#     if count == 0:
#         joint_trajectories_concat = torch.from_numpy(data['trajectories'][i]).float()
#         tau_list_concat = torch.from_numpy(data['torques'][i]).float()
#     else:
#         joint_trajectories_concat = torch.cat((joint_trajectories_concat,torch.from_numpy(data['trajectories'][i]).float()),dim=0)
#         tau_list_concat = torch.cat((tau_list_concat,torch.from_numpy(data['torques'][i]).float()),dim=0)

# TRAJ_val = TensorDataset(joint_trajectories_concat,tau_list_concat)

# create testing dataset
for count, i in enumerate(test_trajectories):
    if count == 0:
        joint_trajectories_concat = torch.from_numpy(data['trajectories'][i]).float()
        tau_list_concat = torch.from_numpy(data['torques'][i]).float()
    else:
        joint_trajectories_concat = torch.cat((joint_trajectories_concat,torch.from_numpy(data['trajectories'][i]).float()),dim=0)
        tau_list_concat = torch.cat((tau_list_concat,torch.from_numpy(data['torques'][i]).float()),dim=0)

TRAJ_test = TensorDataset(joint_trajectories_concat,tau_list_concat)

print("Done!")

# Create dataloaders
trainloader = DataLoader(TRAJ_train, batch_size=192, shuffle=False)
# valloader = DataLoader(TRAJ_val, batch_size=192, shuffle=False)
testloader = DataLoader(TRAJ_test, batch_size=192, shuffle=False)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Design your own network, define layers here.
        # Here We provide a sample of two-layer fully-connected network from HW4 Part3.
        # Your solution, however, should contain convolutional layers.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout
        # If you have many layers, consider using nn.Sequential() to simplify your code
        h1_dim = 6
        h2_dim = 6

        self.fc1 = nn.Linear(6, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc_last = nn.Linear(h2_dim, 2)

    def forward(self,x):
        # TODO: Design your own network, implement forward pass here
        q = x
        x = F.relu(self.fc1(q))
        x = F.relu(self.fc2(x))
        x = self.fc_last(x)
        # The loss layer will be applied outside Network class
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
criterion = nn.MSELoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
num_epoch = 300 # TODO: Choose an appropriate number of training epochs

def train(model, loader, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch.view(batch.shape[0],-1)) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    MSEs = []
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch.view(batch.shape[0],-1))
            MSE_error = criterion(pred, label)
            MSEs.append(MSE_error.item())
            fig, axs = plt.subplots(2, sharex=True)
            # axs[0].plot(label[:,0],label='Calculated',color='b')
            # axs[0].plot(pred[:,0],label='Predicted',color='r')
            # axs[0].legend()
            # axs[0].set_ylabel(r'$\tau_1\,(N-m)$')
            # axs[1].plot(label[:,1],label='Calculated',color='b')
            # axs[1].plot(pred[:,1],label='Predicted',color='r')
            # axs[1].legend()
            # axs[1].set_xlabel('Time Step')
            # axs[1].set_ylabel(r'$\tau_2\,(N-m)$')
            # fig.suptitle('DeLaN Network')
            # plt.show()
            # plt.close()
    Ave_MSE = np.mean(np.array(MSEs))
    print("Average Evaluation MSE: {}".format(Ave_MSE))
    return Ave_MSE

if __name__ == '__main__':
    train(model, trainloader, num_epoch)
    # evaluate(model, valloader)
    evaluate(model, testloader)
