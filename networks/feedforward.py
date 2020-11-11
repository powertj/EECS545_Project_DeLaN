from torch import nn
from torch.nn import functional as F


class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_last = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_last(x)
        return x
