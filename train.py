from networks.delan import DeepLagrangianNetwork
from networks.feedforward import FNN
import numpy as np
from scipy.io import loadmat
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from trajectory_selection import random_train_test_trajectories, select_train_test_trajectories, random_train_test_chars

# torch.manual_seed(0) # Fix random seed for reproducibilit


def train_delan(model, criterion, loader, device, optimizer, scheduler, num_epoch=10):
    print("Start training...")
    model.train()  # Set the model to training mode
    for i in tqdm(range(num_epoch)):
        running_loss = []
        for state, tau, _, _, _, _ in loader:
            state = state.to(device)
            tau = tau.to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            pred_tau, pred_H, pred_c, pred_g = model(state)  # This will call Network.forward() that you implement

            loss = criterion(pred_tau, tau)  # Calculate the loss
            running_loss.append(loss.item())
            loss.backward()  # Backprop gradients to all tensors in the network
            torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step()  # Update trainable weights

        scheduler.step()
        if i % 10 == 0:
            print("Epoch {} loss:{}".format(i + 1, np.mean(running_loss)))  # Print the average loss for this epoch

    print("Done!")


def train_ffn(model, criterion, loader, device, optimizer, scheduler, num_epoch=10):  # Train the model
    print("Start training...")
    model.train()  # Set the model to training mode
    for i in tqdm(range(num_epoch)):
        running_loss = []
        for state, tau, _, _, _, _ in loader:
            state = state.to(device)
            tau = tau.to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            pred = model(state)  # This will call Network.forward() that you implement
            loss = criterion(pred, tau)  # Calculate the loss
            running_loss.append(loss.item())
            loss.backward()  # Backprop gradients to all tensors in the network
            torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step()  # Update trainable weights

        scheduler.step()
        if i % 10 == 0:
            print("Epoch {} loss:{}".format(i + 1, np.mean(running_loss)))  # Print the average loss for this epoch
    print("Done!")


def evaluate_delan(model, criterion, loader, device, show_plots=False,
             num_plots=1):  # Evaluate accuracy on validation / test set
    model.eval()  # Set the model to evaluation mode
    MSEs = []
    i = 0
    with torch.no_grad():  # Do not calculate grident to speed up computation
        for state, tau, g, c, h, label in loader:
            state = state.to(device)
            tau = tau.to(device)
            g = g.to(device)
            c = c.to(device)
            h = h.to(device)
            pred_tau, pred_Hq_ddot, pred_c, pred_g = model(state)

            MSE_error = criterion(pred_tau, tau)
            MSEs.append(MSE_error.item())
            Hq_ddot = (h @ state[:, -2:].unsqueeze(2)).squeeze()
            if show_plots:
                if i < num_plots:
                    fig, axs = plt.subplots(2, 4, figsize=(14.0, 8.0), sharex=True)
                    axs[0, 0].plot(tau[:, 0], label='Calculated', color='b')
                    axs[0, 0].plot(pred_tau[:, 0], label='Predicted', color='r')
                    axs[0, 0].legend()
                    axs[0, 0].set_title(r'$\mathbf{\tau}$')
                    axs[0, 0].set_ylabel('Torque 1 (N-m)')
                    axs[1, 0].plot(tau[:, 1], label='Calculated', color='b')
                    axs[1, 0].plot(pred_tau[:, 1], label='Predicted', color='r')
                    axs[1, 0].set_xlabel('Time Step')
                    axs[1, 0].set_ylabel('Torque 2 (N-m)')
                    axs[0, 1].set_title(r'$\mathbf{H(q)\ddot{q}}$')
                    axs[0, 1].plot(Hq_ddot[:, 0], label='Calculated', color='b')
                    axs[0, 1].plot(pred_Hq_ddot[:, 0], label='Predicted', color='r')
                    axs[1, 1].plot(Hq_ddot[:, 1], label='Calculated', color='b')
                    axs[1, 1].plot(pred_Hq_ddot[:, 1], label='Predicted', color='r')
                    axs[1, 1].set_xlabel('Time Step')
                    axs[0, 2].set_title(r'$\mathbf{c(q,\dot{q})}$')
                    axs[0, 2].plot(c[:, 0], label='Calculated', color='b')
                    axs[0, 2].plot(pred_c[:, 0], label='Predicted', color='r')
                    axs[1, 2].plot(c[:, 1], label='Calculated', color='b')
                    axs[1, 2].plot(pred_c[:, 1], label='Predicted', color='r')
                    axs[1, 2].set_xlabel('Time Step')
                    axs[0, 3].set_title(r'$\mathbf{g(q)}$')
                    axs[0, 3].plot(g[:, 0], label='Calculated', color='b')
                    axs[0, 3].plot(pred_g[:, 0], label='Predicted', color='r')
                    axs[1, 3].plot(g[:, 1], label='Calculated', color='b')
                    axs[1, 3].plot(pred_g[:, 1], label='Predicted', color='r')
                    axs[1, 3].set_xlabel('Time Step')
                    fig.suptitle('Reacher DeLaN Network Trajectory {}'.format(str(label)))
                    plt.show()
                    plt.close()
                    i += 1

    Ave_MSE = np.mean(np.array(MSEs))
    print("Average Evaluation MSE: {}".format(Ave_MSE))
    return Ave_MSE


def evaluate_ffn(model, criterion, loader, device, show_plots=False, num_plots=1): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    MSEs = []
    i = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for state, tau, _, _, _, label in loader:
            state = state.to(device)
            tau = tau.to(device)
            pred = model(state)
            MSE_error = criterion(pred, tau)
            MSEs.append(MSE_error.item())
            if show_plots:
                if i < num_plots:
                    if label == 'a':
                        np.savetxt('reacher_ff_1_char.txt', np.concatenate((tau,pred),axis=1))
                    fig, axs = plt.subplots(2, sharex=True)
                    axs[0].plot(tau[:,0],label='Calculated',color='b')
                    axs[0].plot(pred[:,0],label='Predicted',color='r')
                    axs[0].legend()
                    axs[0].set_ylabel(r'$\tau_1\,(N-m)$')
                    axs[1].plot(tau[:,1],label='Calculated',color='b')
                    axs[1].plot(pred[:,1],label='Predicted',color='r')
                    axs[1].set_xlabel('Time Step')
                    axs[1].set_ylabel(r'$\tau_2\,(N-m)$')
                    fig.suptitle('Reacher FF-NN Trajectory {}'.format(str(label)))
                    plt.show()
                    plt.close()
                    i += 1

    Ave_MSE = np.mean(np.array(MSEs))
    print("Average Evaluation MSE: {}".format(Ave_MSE))
    return Ave_MSE


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="Reacher", choices=["Reacher", "Cartpole"])
    parser.add_argument("--model", type=str, default="delan", choices=["delan", "feedforward"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=64)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()

    # Load the dataset and train and test splits
    print("Loading dataset...")

    if args.experiment == "Reacher":
        q_dim = 2
        action_dim = 2
        data = np.load('data/trajectories_joint_space.npz', allow_pickle=True)

        train_trajectories, train_labels, test_trajectories, test_labels = random_train_test_chars(data,
                                                                                               num_train_chars=15,
                                                                                               num_samples_per_char=1)

        print("Done!")
    elif args.experiment == "Cartpole":
        q_dim = 2
        action_dim = 2

        fname = 'cartpole_traj_gen/data/cartpole_all_200hz.mat'
        data = loadmat(fname)
        train_trajectories, \
        train_labels, \
        test_trajectories, \
        test_labels = select_train_test_trajectories(data, train_label_types=[1, 2, 4], num_samples_per_label=5)
    else:
        raise NotImplementedError

    TRAJ_train = TrajectoryDataset(data, train_trajectories, train_labels)
    TRAJ_test = TrajectoryDataset(data, test_trajectories, test_labels)
    trainloader = DataLoader(TRAJ_train, batch_size=None)
    testloader = DataLoader(TRAJ_test, batch_size=None)

    device = args.device

    if args.model == "delan":
        model = DeepLagrangianNetwork(q_dim, args.hidden_size, device=device).to(device)
        evaluate = evaluate_delan
        train = train_delan
    elif args.model == "feedforward":
        model = FNN(q_dim * 3, action_dim, args.hidden_size).to(device)
        evaluate = evaluate_ffn
        train = train_ffn
    else:
        raise NotImplementedError

    criterion = nn.MSELoss()  # Specify the loss layer
    # Modify the line below, experiment with different optimizers and parameters (such as learning rate)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)  # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    # train and evaluate network
    train(model, criterion, trainloader, device, optimizer, scheduler, args.num_epochs)
    evaluate(model, criterion, testloader, device, show_plots=False)

