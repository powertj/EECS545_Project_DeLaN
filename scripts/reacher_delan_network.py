import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader, random_split

# Load the dataset and train, val, test splits
print("Loading datasets...")

data = np.load('../data/trajectories_joint_space.npz',allow_pickle=True)

train_characters = range(2)
val_characters = range(20,25)
test_characters = range(25,30)

# create training dataset
for count, i in enumerate(train_characters):
    if count == 0:
        joint_trajectories_concat = torch.from_numpy(data['trajectories'][i]).float()
        tau_list_concat = torch.from_numpy(data['torques'][i]).float()
    else:
        joint_trajectories_concat = torch.cat((joint_trajectories_concat,torch.from_numpy(data['trajectories'][i]).float()),dim=0)
        tau_list_concat = torch.cat((tau_list_concat,torch.from_numpy(data['torques'][i]).float()),dim=0)

TRAJ_train = TensorDataset(joint_trajectories_concat,tau_list_concat)

# create validation dataset
for count, i in enumerate(val_characters):
    if count == 0:
        joint_trajectories_concat = torch.from_numpy(data['trajectories'][i]).float()
        tau_list_concat = torch.from_numpy(data['torques'][i]).float()
    else:
        joint_trajectories_concat = torch.cat((joint_trajectories_concat,torch.from_numpy(data['trajectories'][i]).float()),dim=0)
        tau_list_concat = torch.cat((tau_list_concat,torch.from_numpy(data['torques'][i]).float()),dim=0)

TRAJ_val = TensorDataset(joint_trajectories_concat,tau_list_concat)

# create testing dataset
for count, i in enumerate(test_characters):
    if count == 0:
        joint_trajectories_concat = torch.from_numpy(data['trajectories'][i]).float()
        tau_list_concat = torch.from_numpy(data['torques'][i]).float()
    else:
        joint_trajectories_concat = torch.cat((joint_trajectories_concat,torch.from_numpy(data['trajectories'][i]).float()),dim=0)
        tau_list_concat = torch.cat((tau_list_concat,torch.from_numpy(data['torques'][i]).float()),dim=0)

TRAJ_test = TensorDataset(joint_trajectories_concat,tau_list_concat)

print("Done!")

# Create dataloaders
trainloader = DataLoader(TRAJ_train, batch_size=192, shuffle=True)
valloader = DataLoader(TRAJ_val, batch_size=192, shuffle=True)
testloader = DataLoader(TRAJ_test, batch_size=192, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Design your own network, define layers here.
        input_dim = 2
        h1_dim = 6
        # joint angle input
        self.fc1 = nn.Linear(input_dim, h1_dim)
        # gravity layer
        self.fc2 = nn.Linear(h1_dim, input_dim) 
        # ld layer
        self.fc3 = nn.Linear(h1_dim, input_dim)
        # lo layer
        self.fc4 = nn.Linear(h1_dim, 1)

    def forward(self,x):
        # TODO: Design your own network, implement forward pass here
        d = x.shape[1] // 3
        n = x.shape[0]
        q, q_dot, q_ddot = torch.split(x,[d,d,d], dim = 1)

        h1 = F.relu(self.fc1(q))
        # Gravity torque
        g = self.fc2(h1)

        # ld is vector of diagonal L terms, lo is vector of off-diagonal L terms
        ld = F.relu(self.fc3(h1))
        lo = self.fc4(h1)

        dRelu_fc1 = torch.where(h1 > 0, torch.ones(h1.shape), torch.zeros(h1.shape))
        dh1_dq = torch.diag_embed(dRelu_fc1) @ self.fc1.weight.T.T

        dRelu_fc3 = torch.where(ld > 0, torch.ones(ld.shape), torch.zeros(ld.shape))
        dld_dh1 = torch.diag_embed(dRelu_fc3) @ self.fc3.weight.T.T
        dlo_dh1 = self.fc4.weight.T.T

        # dld_dh1 = torch.cat([dld_dh1,torch.zeros(n,d,self.fc4.weight.T.shape[0])],dim=2)
        # dlo_dh1 = torch.cat([torch.zeros(n,1,d),torch.stack(n * [self.fc4.weight.T.T])],dim=2)
        # dh2_dh1 = torch.cat([dh2_dh1,torch.cat([torch.zeros(n,1,d),torch.stack(n * [self.fc4.weight.T.T])],dim=2)],dim=1)
        # dl_dq = dh2_dh1 @ dh1_dq
        
        dld_dq = dld_dh1 @ dh1_dq
        dlo_dq = dlo_dh1 @ dh1_dq
        dld_dqi = dld_dq.permute(0,2,1).view(n,d,d,1)
        dlo_dqi = dlo_dq.permute(0,2,1).view(n,d,1,1)

        # dl_dq = torch.cat([dld_dq,dlo_dq],dim=1)

        dld_dt = dld_dq @ q_dot.view(n,d,1)
        dlo_dt = dlo_dq @ q_dot.view(n,d,1)

        dL_dt = torch.tril(torch.ones(n,d,d)) - torch.eye(d)
        dL_dqi = torch.tril(torch.ones(n,d,d,d)) - torch.eye(d)

        L = torch.tril(torch.ones(n,d,d)) - torch.eye(d)
        indices = dL_dt == 1
        indices_dL_dqi = dL_dqi == 1

        dL_dt[indices] = dlo_dt.view(n)
        L[indices] = lo.view(n)
        dL_dqi[indices_dL_dqi] = dlo_dqi.view(n*d)

        dL_dt += torch.diag_embed(dld_dt.view(n,d))
        L += torch.diag_embed(ld.view(n,d))
        dL_dqi += torch.diag_embed(dld_dqi.view(n,d,d))

        # Mass Matrix
        epsilon = .001
        H = L.permute(0,2,1) @ L + epsilon * torch.eye(d)

        # Time derivative of Mass Matrix
        dH_dt = L @ dL_dt.permute(0,2,1) + dL_dt @ L.permute(0,2,1)

        quadratic_term = q_dot.view(n,1,1,d) @ (dL_dqi @ L.permute(0,2,1).view(n,1,d,d) + L.view(n,1,d,d) @ dL_dqi.permute(0,1,3,2)) @ q_dot.view(n,1,d,1)
        tau =  H @ q_ddot.view(n,d,1) + dH_dt @ q_dot.view(n,d,1) + quadratic_term.view(n,d,1) + g.view(n,d,1)

        # The loss layer will be applied outside Network class
        return tau.squeeze()

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
criterion = nn.MSELoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-4) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
num_epoch = 50 # TODO: Choose an appropriate number of training epochs

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
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch.view(batch.shape[0],-1))
            MSE_error = criterion(pred, label)
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
            fig.suptitle('DeLaN Network')
            # plt.show()
            plt.close()

    print("Evaluation MSE: {}".format(MSE_error))
    return MSE_error

if __name__ == '__main__':
    train(model, trainloader, num_epoch)
    evaluate(model, valloader)
    evaluate(model, testloader)
