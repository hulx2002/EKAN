import torch
import torch.nn as nn
import numpy as np
from spline import *
from representation import *

class EKANLayer(nn.Module):
    def __init__(self, rep_in, rep_out, num=5, k=3, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], device='cpu'):
        super(EKANLayer, self).__init__()

        gated_rep_in = gated(rep_in)
        rep_in_multi = rep_in * (num + k + 1)
        gated_rep_out = gated(rep_out)
        self.rep_in = rep_in
        self.rep_out = rep_out
        self.gated_rep_in = gated_rep_in
        self.rep_in_multi = rep_in_multi
        self.gated_rep_out = gated_rep_out

        in_dim = rep_in.size()
        out_dim = rep_out.size()
        gated_in_dim = gated_rep_in.size()
        in_multi_dim = rep_in_multi.size()
        gated_out_dim = gated_rep_out.size()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gated_in_dim = gated_in_dim
        self.in_multi_dim = in_multi_dim
        self.gated_out_dim = gated_out_dim
        self.num = num
        self.k = k

        u, indices = np.unique(gate_indices(rep_in), return_inverse=True)
        self.u = u
        self.indices = indices
        self.grid = torch.einsum('i,j->ij', torch.ones(len(u), device=device), torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        self.grid = nn.Parameter(self.grid).requires_grad_(False)

        self.base_fun = base_fun

        self.grid_eps = grid_eps
        self.device = device

        self.linear = nn.Linear(in_multi_dim, gated_out_dim, device=device)
        nn.init.orthogonal_(self.linear.weight)
        rep_Wi = gated_rep_out * rep_in.T
        Pwi_terms, multiplicities, perm, invperm = rep_Wi.equivariant_projector(lazy=True)
        Pwi_terms = [torch.tensor(Pwi_term, dtype=torch.float32, device=device) for Pwi_term in Pwi_terms]
        self.Pwi = lambda w: lazy_P(Pwi_terms, multiplicities, perm, invperm, w)
        self.Pwi_inv = lambda w: lazy_Pinv(Pwi_terms, multiplicities, perm, invperm, w)
        self.Pb = torch.tensor(gated_rep_out.equivariant_projector(), dtype=torch.float32, device=device)

        Wdim, bi_weight_proj = bilinear_weights(gated_rep_out, gated_rep_out)
        self.bi_weight_proj = bi_weight_proj
        self.bi_w = nn.Parameter(torch.randn(Wdim, device=device))
    
    def weight_proj(self):
        w_reshape = self.linear.weight.reshape(self.gated_out_dim, self.num + self.k + 1, self.in_dim).permute(0, 2, 1).reshape(self.gated_out_dim * self.in_dim, self.num + self.k + 1)
        W_reshape = self.Pwi(w_reshape)
        W = W_reshape.reshape(self.gated_out_dim, self.in_dim, self.num + self.k + 1).permute(0, 2, 1).reshape(self.gated_out_dim, (self.num + self.k + 1) * self.in_dim)
        return W

    def forward(self, x):
        x = x.to(self.device)
        batch = x.shape[0]
        gate_scalars = x[..., self.u].permute(1, 0)
        gate_scalars_sp = B_batch(x=gate_scalars, grid=self.grid, k=self.k, extend=True, device=self.device).permute(2, 1, 0)
        gate_scalars_sb = self.base_fun(gate_scalars).permute(1, 0).unsqueeze(dim=1)
        gate_scalars = torch.cat((gate_scalars_sp, gate_scalars_sb), dim=1)
        gate_scalars = gate_scalars[..., self.indices]
        activations = torch.einsum('ijk,ik->ijk', gate_scalars, x[..., : self.in_dim]).reshape(batch, -1)
        W = self.weight_proj()
        b = self.Pb @ self.linear.bias
        out = activations @ W.T + b

        bi_W = self.bi_weight_proj(self.bi_w, out, self.device)
        out = .1 * (bi_W @ out[..., None])[..., 0] + out
        return out

    def weight_proj_inv(self, W):
        W_reshape = W.reshape(self.gated_out_dim, self.num + self.k + 1, self.in_dim).permute(0, 2, 1).reshape(self.gated_out_dim * self.in_dim, self.num + self.k + 1)
        w_reshape = self.Pwi_inv(W_reshape)
        self.linear.weight.data = w_reshape.reshape(self.gated_out_dim, self.in_dim, self.num + self.k + 1).permute(0, 2, 1).reshape(self.gated_out_dim, (self.num + self.k + 1) * self.in_dim)

    def update_grid_from_samples(self, x):
        x = x.to(self.device)
        batch = x.shape[0]

        gate_scalars = x[..., self.u].permute(1, 0)
        x_pos = torch.sort(gate_scalars, dim=1)[0]
        gate_scalars_sp = B_batch(x=gate_scalars, grid=self.grid, k=self.k, extend=True, device=self.device).permute(2, 1, 0)
        gate_scalars_sb = self.base_fun(gate_scalars).permute(1, 0).unsqueeze(dim=1)
        gate_scalars = torch.cat((gate_scalars_sp, gate_scalars_sb), dim=1)
        gate_scalars = gate_scalars[..., self.indices]
        activations = torch.einsum('ijk,ik->ijk', gate_scalars, x[..., : self.in_dim]).reshape(batch, -1)
        W = self.weight_proj()
        y_eval = activations @ W.T

        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat([grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        gate_scalars_update = x[..., self.u].permute(1, 0)
        gate_scalars_sp_update = B_batch(x=gate_scalars_update, grid=self.grid, k=self.k, extend=True, device=self.device).permute(2, 1, 0)
        gate_scalars_sb_update = self.base_fun(gate_scalars_update).permute(1, 0).unsqueeze(dim=1)
        gate_scalars_update = torch.cat((gate_scalars_sp_update, gate_scalars_sb_update), dim=1)
        gate_scalars_update = gate_scalars_update[..., self.indices]
        activations_update = torch.einsum('ijk,ik->ijk', gate_scalars_update, x[..., : self.in_dim]).reshape(batch, -1)

        W_update = y_eval.T @ torch.linalg.pinv(activations_update.T)
        self.weight_proj_inv(W_update)