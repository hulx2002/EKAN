# modified from https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/nn/pytorch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from representation import *
from LBFGS import *
import os
import glob
from tqdm import tqdm
import random
from adan import Adan

class Linear(nn.Linear):
    def __init__(self, repin, repout, device):
        nin, nout = repin.size(), repout.size()
        super().__init__(nin, nout, device=device)
        nn.init.orthogonal_(self.weight)
        rep_W = repout * repin.T
        rep_bias = repout
        Pw_terms, multiplicities, perm, invperm = rep_W.equivariant_projector(lazy=True)
        Pw_terms = [torch.tensor(Pw_term, dtype=torch.float32, device=device) for Pw_term in Pw_terms]
        Pb = torch.tensor(rep_bias.equivariant_projector(), dtype=torch.float32, device=device)
        self.proj_b = lambda b: Pb @ b
        self.proj_w = lambda w: lazy_P(Pw_terms, multiplicities, perm, invperm, w.reshape(-1)).reshape(nout, nin)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        return F.linear(x, self.proj_w(self.weight), self.proj_b(self.bias))

class BiLinear(nn.Module):
    def __init__(self, repin, repout, device):
        super().__init__()
        Wdim, weight_proj = bilinear_weights(repout,repin)
        self.weight_proj = weight_proj
        self.bi_params = nn.Parameter(torch.randn(Wdim, device=device))
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        W = self.weight_proj(self.bi_params, x, self.device)
        out = .1 * (W @ x[..., None])[..., 0]
        return out

class GatedNonlinearity(nn.Module):
    def __init__(self, rep, device):
        super().__init__()
        self.rep = rep
        self.device = device

    def forward(self, values):
        values = values.to(self.device)
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = gate_scalars.sigmoid() * values[..., : self.rep.size()]
        return activations

class EMLPBlock(nn.Module):
    def __init__(self, rep_in, rep_out, device):
        super().__init__()
        self.linear = Linear(rep_in, gated(rep_out), device)
        self.bilinear = BiLinear(gated(rep_out), gated(rep_out), device)
        self.nonlinearity = GatedNonlinearity(rep_out, device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        lin = self.linear(x)
        preact = self.bilinear(lin) + lin
        return self.nonlinearity(preact)

class EMLP(nn.Module):
    def __init__(self, rep_in, rep_out, group, width=None, device='cpu', seed=0, classify=False):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group

        middle_layers = [(c(group) if isinstance(c, Rep) else uniform_rep(c, group)) for c in width]
        reps = [self.rep_in] + middle_layers
        self.network = nn.Sequential(
            *[EMLPBlock(rin, rout, device) for rin, rout in zip(reps, reps[1 :])],
            Linear(reps[-1], self.rep_out, device)
        )

        self.device = device
        self.classify = classify

    def forward(self, x):
        x = x.to(self.device)
        if self.classify:
            return torch.sigmoid(self.network(x))
        else:
            return self.network(x)

    def train(self, dataset, opt="Adan", steps=100, log=1, loss_fn=None, lr=1., batch=-1, metrics=None, sglr_avoid=False, device='cpu'):
        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            if self.classify:
                epsilon = 1e-7
                loss_fn = lambda x, y: -torch.mean(y.unsqueeze(1) * torch.log(torch.clamp(x, epsilon, 1 - epsilon)) + (1 - y.unsqueeze(1)) * torch.log(1 - torch.clamp(x, epsilon, 1 - epsilon)))
                loss_fn_eval = lambda x, y: torch.mean(y.unsqueeze(1) * torch.round(x) + (1 - y.unsqueeze(1)) * (1 - torch.round(x)))
            else:
                loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == 'Adan':
            optimizer = Adan(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
        else:
            batch_size = batch
        if batch == -1 or batch > dataset['test_input'].shape[0]:
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size_test = batch

        global train_loss

        def closure():
            global train_loss
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
            else:
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
            objective = train_loss
            objective.backward()
            return objective

        for _ in pbar:

            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam" or opt == 'Adan':
                pred = self.forward(dataset['train_input'][train_id].to(device))
                if sglr_avoid == True:
                    id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                    train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
                else:
                    train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
                loss = train_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))

            if _ % log == 0:
                pbar.set_description("train loss: %.2e | test loss: %.2e " % (train_loss.cpu().detach().numpy(), test_loss.cpu().detach().numpy()))

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(train_loss.cpu().detach().numpy())
            results['test_loss'].append(test_loss.cpu().detach().numpy())

        with torch.no_grad():
            overall_test_loss = 0.
            for i in range(0, dataset['test_input'].shape[0], batch_size_test):
                test_id = np.arange(i, min(i + batch_size_test, dataset['test_input'].shape[0]))
                overall_test_loss += len(test_id) * loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))
            overall_test_loss /= dataset['test_input'].shape[0]
            print(f'overall test loss: {overall_test_loss}')

        return results

    def clear_ckpts(self, folder='./model_ckpt'):
        if os.path.exists(folder):
            files = glob.glob(folder + '/*')
            for f in files:
                os.remove(f)
        else:
            os.makedirs(folder)

    def save_ckpt(self, name, folder='./model_ckpt'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.state_dict(), folder + '/' + name)
        print('save this model to', folder + '/' + name)

    def load_ckpt(self, name, folder='./model_ckpt'):
        self.load_state_dict(torch.load(folder + '/' + name, weights_only=True))