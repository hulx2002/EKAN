# modified from https://github.com/KindXiaoming/pykan/blob/master/kan/KANLayer.py

import torch
import torch.nn as nn
import numpy as np
from spline import *
from representation import *

class KANLayer(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], device='cpu'):
        super(KANLayer, self).__init__()

        in_multi_dim = in_dim * (num + k + 1)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_multi_dim = in_multi_dim
        self.num = num
        self.k = k

        self.grid = torch.einsum('i,j->ij', torch.ones(in_dim, device=device), torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        self.grid = nn.Parameter(self.grid).requires_grad_(False)

        self.base_fun = base_fun

        self.grid_eps = grid_eps
        self.device = device

        self.linear = nn.Linear(in_multi_dim, out_dim, device=device)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x):
        x = x.to(self.device)
        batch = x.shape[0]
        gate_scalars = x.permute(1, 0)
        gate_scalars_sp = B_batch(x=gate_scalars, grid=self.grid, k=self.k, extend=True, device=self.device).permute(2, 1, 0)
        gate_scalars_sb = self.base_fun(gate_scalars).permute(1, 0).unsqueeze(dim=1)
        gate_scalars = torch.cat((gate_scalars_sp, gate_scalars_sb), dim=1)
        activations = torch.einsum('ijk,ik->ijk', gate_scalars, x).reshape(batch, -1)
        y = activations @ self.linear.weight.T + self.linear.bias

        return y

    def update_grid_from_samples(self, x):
        x = x.to(self.device)
        batch = x.shape[0]

        gate_scalars = x.permute(1, 0)
        x_pos = torch.sort(gate_scalars, dim=1)[0]
        gate_scalars_sp = B_batch(x=gate_scalars, grid=self.grid, k=self.k, extend=True, device=self.device).permute(2, 1, 0)
        gate_scalars_sb = self.base_fun(gate_scalars).permute(1, 0).unsqueeze(dim=1)
        gate_scalars = torch.cat((gate_scalars_sp, gate_scalars_sb), dim=1)
        activations = torch.einsum('ijk,ik->ijk', gate_scalars, x).reshape(batch, -1)
        y_eval = activations @ self.linear.weight.T

        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat([grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        gate_scalars_update = x.permute(1, 0)
        gate_scalars_sp_update = B_batch(x=gate_scalars_update, grid=self.grid, k=self.k, extend=True, device=self.device).permute(2, 1, 0)
        gate_scalars_sb_update = self.base_fun(gate_scalars_update).permute(1, 0).unsqueeze(dim=1)
        gate_scalars_update = torch.cat((gate_scalars_sp_update, gate_scalars_sb_update), dim=1)
        activations_update = torch.einsum('ijk,ik->ijk', gate_scalars_update, x).reshape(batch, -1)

        self.linear.weight.data = y_eval.T @ torch.linalg.pinv(activations_update.T)