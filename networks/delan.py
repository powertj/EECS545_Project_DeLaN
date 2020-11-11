import torch
from torch import nn
import torch.nn.functional as F


class DeepLagrangianNetwork(nn.Module):

    def __init__(self, q_dim, hidden_dim=64, device="cpu"):
        super().__init__()
        self.q_dim = q_dim
        self.num_Lo = int(0.5 * (q_dim ** 2 - q_dim))
        self.fc1 = nn.Linear(q_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        self.fc_G = nn.Linear(hidden_dim, q_dim)
        self.fc_Ld = nn.Linear(hidden_dim, q_dim)
        self.fc_Lo = nn.Linear(hidden_dim, self.num_Lo)

        self.act_fn = F.leaky_relu
        self.neg_slope = -0.01
        self.device = device
        self.interim_values = {}

    def compute_gradients_for_forward_pass(self, qdot, h1, h2, h3):
        """
        Computes partial derivatives of the inertia matrix (H) needed for the forward pass
        :return: dHdq and dHdt
        """
        n, d = qdot.shape

        dRelu_fc1 = torch.where(h1 > 0, torch.ones(h1.shape, device=self.device),
                                self.neg_slope * torch.ones(h1.shape, device=self.device))
        dh1_dq = torch.diag_embed(dRelu_fc1) @ self.fc1.weight

        dRelu_fc2 = torch.where(h2 > 0, torch.ones(h2.shape, device=self.device),
                                 self.neg_slope * torch.ones(h2.shape, device=self.device))
        dh2_dh1 = torch.diag_embed(dRelu_fc2) @ self.fc2.weight

        dRelu_dfc_Ld = torch.sigmoid(h3)  # torch.where(ld > 0, torch.ones(ld.shape), 0.0 * torch.ones(ld.shape))

        dld_dh2 = torch.diag_embed(dRelu_dfc_Ld) @ self.fc_Ld.weight
        dlo_dh2 = self.fc_Lo.weight

        dld_dq = dld_dh2 @ dh2_dh1 @ dh1_dq
        dlo_dq = dlo_dh2 @ dh2_dh1 @ dh1_dq

        dld_dt = (dld_dq @ qdot.view(n, d, 1)).squeeze(-1)
        dlo_dt = (dlo_dq @ qdot.view(n, d, 1)).squeeze(-1)
        dld_dq = dld_dq.permute(0, 2, 1)

        dL_dt = self.assemble_lower_triangular_matrix(dlo_dt, dld_dt)
        dL_dq = self.assemble_lower_triangular_matrix(dlo_dq, dld_dq)

        return dL_dq, dL_dt

    def assemble_lower_triangular_matrix(self, Lo, Ld):
        """
        Assembled a lower triangular matrix from it's diagonal and off-diagonal elements
        :param Lo: Off diagonal elements of lower triangular matrix
        :param Ld: Diagonal elements of lower triangular matrix
        :return: Lower triangular matrix L
        """
        assert (2 * Lo.shape[1] == (Ld.shape[1]**2 - Ld.shape[1]))

        diagonal_matrix = torch.diag_embed(Ld)
        L = torch.tril(torch.ones(*diagonal_matrix.shape, device=self.device)) - torch.eye(self.q_dim)

        # Set off diagonals
        L[L==1] = Lo.view(-1)
        # Add diagonals
        L = L + diagonal_matrix
        return L

    def forward(self, x):
        """
        Deep Lagrangian Network inverse action model forward pass
        :param x: State consisting of q, q_dot, q_ddot
        :return: tau - action, H @ q_ddot, C, G

        where H is inertia matrix, C coriolis term, G is potentials term

        """
        q, q_dot, q_ddot = torch.chunk(x, chunks=3, dim=1)
        n, d = q.shape

        hidden1 = self.act_fn(self.fc1(q))
        hidden2 = self.act_fn(self.fc2(hidden1))
        hidden3 = self.fc_Ld(hidden2)

        g = self.fc_G(hidden2)
        Ld = F.softplus(hidden3)
        Lo = self.fc_Lo(hidden2)
        L = self.assemble_lower_triangular_matrix(Lo, Ld)
        dL_dq, dL_dt = self.compute_gradients_for_forward_pass(q_dot, hidden1, hidden2, hidden3)

        # Inertia matrix and time derivative
        H = L @ L.transpose(1, 2) + 1e-9 * torch.eye(d, device=self.device)
        dH_dt = L @ dL_dt.permute(0, 2, 1) + dL_dt @ L.permute(0, 2, 1)

        # Compute quadratic term d/dq [q_dot.T @ H @ q_dot]
        q_dot_repeated = q_dot.repeat(d, 1)
        dL_dqi = dL_dq.view(n * d, d, d)
        L_repeated = L.repeat(d, 1, 1)
        quadratic_term = q_dot_repeated.view(-1, 1, d) @ (dL_dqi @ L_repeated.transpose(1, 2) +
                                           L_repeated @ dL_dqi.transpose(1, 2)) @ q_dot_repeated.view(-1, d, 1)

        # Compute coriolis term
        c = dH_dt @ q_dot.view(n, d, 1) - 0.5 * quadratic_term.view(n, d, 1)
        tau = H @ q_ddot.view(n, d, 1) + c + g.view(n, d, 1)

        return tau.squeeze(), (H @ q_ddot.view(n, d, 1)).squeeze(), c.squeeze(), g.squeeze()


if __name__ == "__main__":

    network = DeepLagrangianNetwork(2, 64)

    test_input = torch.ones(1, 6)
    network(test_input)
