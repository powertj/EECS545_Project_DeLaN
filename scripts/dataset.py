import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, data, indices):
        self.indices = indices
        self.trajectories = data['trajectories']
        self.torques = data['torques']
        self.g = data['g']
        self.H = data['H']

        if 'c' in data:
            self.c = data['c']
        else:
            self.c = np.zeros(self.torques.shape)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trajTensor = torch.from_numpy(self.trajectories[self.indices[idx]]).float()
        torqueTensor = torch.from_numpy(self.torques[self.indices[idx]]).float()
        gTensor = torch.from_numpy(self.g[self.indices[idx]]).float()
        cTensor = torch.from_numpy(self.c[self.indices[idx]]).float()
        HTensor = torch.from_numpy(self.H[self.indices[idx]]).float()


        return (trajTensor, torqueTensor, gTensor, cTensor, HTensor)
