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
from dataset import TrajectoryDataset
# torch.manual_seed(0) # Fix random seed for reproducibility

def generate_train_test_indices(data, num_train_chars=1, num_samples_per_char=1):
    char_count = {}
    char_indices = {}
    train_chars = []
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

    for i in range(num_train_chars):
        if len(test_chars) > 0:
            train_char_idx = random.randint(0,len(test_chars)-1)
            train_char = test_chars.pop(train_char_idx)
            train_chars.append(train_char)
            if num_samples_per_char < len(char_indices[train_char]):
                train_trajectories += char_indices[train_char][:num_samples_per_char]
            else:
                train_trajectories += char_indices[train_char]

    for test_char in test_chars:
            if num_samples_per_char < len(char_indices[test_char]):
                test_trajectories += char_indices[test_char][:num_samples_per_char]
            else:
                test_trajectories += char_indices[test_char]

    return train_trajectories, test_trajectories


class Reacher_DeLaN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Design your own network, define layers here.
        input_dim = 2
        h1_dim = 64
        h2_dim = 64
        # joint angle input layer
        self.fc1 = nn.Linear(input_dim, h1_dim)
        #self.fc1b = nn.Linear(h1_dim, h1_dim)
        # torch.nn.init.zeros_(self.fc1.weight)
        # torch.nn.init.zeros_(self.fc1.bias)

        # 1st hidden layer
        self.fc1a = nn.Linear(h1_dim, h2_dim)
        # torch.nn.init.zeros_(self.fc1a.weight)
        # torch.nn.init.zeros_(self.fc1a.bias)

        # gravity layer
        self.fc2 = nn.Linear(h2_dim, input_dim) 
        # torch.nn.init.zeros_(self.fc2.weight)
        # torch.nn.init.zeros_(self.fc2.bias)

        # ld layer
        self.fc3 = nn.Linear(h2_dim, input_dim)
        # torch.nn.init.ones_(self.fc3.weight)
        # torch.nn.init.ones_(self.fc3.bias)

        # lo layer
        self.fc4 = nn.Linear(h2_dim, 1)
        # torch.nn.init.zeros_(self.fc4.weight)
        # torch.nn.init.zeros_(self.fc4.bias)

        self.act_fn = F.leaky_relu
        self.neg_slope = -0.01


    def forward(self,x):
        d = x.shape[1] // 3
        num_off_diagonals = d * (d - 1) // 2
        n = x.shape[0]
        q, q_dot, q_ddot = torch.split(x,[d,d,d], dim = 1)

        # q.requires_grad = True
        h1 = self.act_fn(self.fc1(q))
        h2 = self.act_fn(self.fc1a(h1))
        
        # Gravity torque
        g = self.fc2(h2)

        # ld is vector of diagonal L terms, lo is vector of off-diagonal L terms
        ld = F.relu(self.fc3(h2))
        lo = self.fc4(h2)

        dRelu_fc1 = torch.where(h1 > 0, torch.ones(h1.shape), self.neg_slope * torch.ones(h1.shape))
        dh1_dq = torch.diag_embed(dRelu_fc1) @ self.fc1.weight

        dRelu_fc1a = torch.where(h2 > 0, torch.ones(h2.shape), self.neg_slope * torch.ones(h2.shape))
        dh2_dh1 = torch.diag_embed(dRelu_fc1a) @ self.fc1a.weight

        dRelu_fc3 = torch.where(ld > 0, torch.ones(ld.shape), 0.0 * torch.ones(ld.shape))

        dld_dh2 = torch.diag_embed(dRelu_fc3) @ self.fc3.weight
        dlo_dh2 = self.fc4.weight
        
        dld_dq = dld_dh2 @ dh2_dh1 @ dh1_dq
        dlo_dq = dlo_dh2 @ dh2_dh1 @ dh1_dq
        dld_dqi = dld_dq.permute(0,2,1).view(n,d,d,1)
        dlo_dqi = dlo_dq.permute(0,2,1).view(n,d,-1,1)

        # dl_dq = torch.cat([dld_dq,dlo_dq],dim=1)

        dld_dt = dld_dq @ q_dot.view(n,d,1)
        dlo_dt = dlo_dq @ q_dot.view(n,d,1)

        # Get L, dL matrices without inplace operations
        L = []
        dL_dt = []
        dL_dqi = []
        zeros = torch.zeros_like(ld)
        zeros_2 = torch.zeros_like(dld_dqi)
        lo_start = 0
        lo_end = d - 1
        for i in range(d):
            l = torch.cat((zeros[:, :i].view(n, -1), ld[:, i].view(-1, 1), lo[:, lo_start:lo_end]), dim=1)
            dl_dt = torch.cat((zeros[:, :i].view(n, -1), dld_dt[:, i].view(-1, 1),
                               dlo_dt[:, lo_start:lo_end].view(n, -1)), dim=1)

            dl_dqi = torch.cat((zeros_2[:, :, :i].view(n, d, -1), dld_dqi[:, :, i].view(n, -1, 1),
                                dlo_dqi[:, :, lo_start:lo_end].view(n, d, -1)), dim=2)

            lo_start = lo_start + lo_end
            lo_end = lo_end + d - 2 - i
            L.append(l)
            dL_dt.append(dl_dt)
            dL_dqi.append(dl_dqi)

        L = torch.stack(L, dim=2)
        dL_dt = torch.stack(dL_dt, dim=2)

        # dL_dqi n x d x d x d -- last dim is index for qi
        dL_dqi = torch.stack(dL_dqi, dim=3).permute(0, 2, 3, 1)

        epsilon = .00001    #small number to ensure positive definiteness of H

        H = L @ L.transpose(1, 2) + epsilon * torch.eye(d)

        # Time derivative of Mass Matrix
        dH_dt = L @ dL_dt.permute(0,2,1) + dL_dt @ L.permute(0,2,1)

        quadratic_term = []
        for i in range(d):
            qterm = q_dot.view(n, 1, d) @ (dL_dqi[:, :, :, i] @ L.transpose(1, 2) +
                                           L @ dL_dqi[:, :, :, i].transpose(1, 2)) @ q_dot.view(n, d, 1)
            quadratic_term.append(qterm)

        quadratic_term = torch.stack(quadratic_term, dim=1)

        c = dH_dt @ q_dot.view(n,d,1) - 0.5 * quadratic_term.view(n,d,1)

        tau = H @ q_ddot.view(n,d,1) + c + g.view(n,d,1)

        # The loss layer will be applied outside Network class
        return (tau.squeeze(), H.squeeze(), c.squeeze(), g.squeeze())


def train(model, loader, num_epoch, optimizer, scheduler): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred_tau, pred_H, pred_c, pred_g = model(batch) # This will call Network.forward() that you implement

            loss = criterion(pred_tau, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            #print(model.fc4.weight.grad)
            torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step() # Update trainable weights

        scheduler.step()
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    
    print("Done!")


def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    MSEs = []
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred_tau, pred_H, pred_c, pred_g = model(batch)
            MSE_error = criterion(pred_tau, label)
            MSEs.append(MSE_error.item())
            # fig, axs = plt.subplots(2, sharex=True)
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

    # Load the dataset and train and test splits
    print("Loading dataset...")
    data = np.load('../data/trajectories_joint_space.npz', allow_pickle=True)
    train_trajectories, test_trajectories = generate_train_test_indices(data, num_train_chars=2, num_samples_per_char=2)
    TRAJ_train = TrajectoryDataset(data,train_trajectories)
    TRAJ_test = TrajectoryDataset(data,test_trajectories)

    print("Done!")
    trainloader = DataLoader(TRAJ_train, batch_size=None)
    testloader = DataLoader(TRAJ_test, batch_size=None)

    # create model and specify hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
    model = Reacher_DeLaN_Network().to(device)
    criterion = nn.MSELoss() # Specify the loss layer
    # TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=0.1) #Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    num_epoch = 200 # TODO: Choose an appropriate number of training epochs


    # train and evaluate network
    train(model, trainloader, num_epoch, optimizer, scheduler)
    evaluate(model, testloader)
