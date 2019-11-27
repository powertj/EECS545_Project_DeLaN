import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, data, indices):
        self.indices = indices
        self.keys = data['keys']
        self.trajectories = data['trajectories']
        self.torques = data['torques']
        self.labels = data['labels']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trajTensor = torch.from_numpy(self.trajectories[self.indices[idx]]).float()
        torqueTensor = torch.from_numpy(self.torques[self.indices[idx]]).float()

        return (trajTensor, torqueTensor)
