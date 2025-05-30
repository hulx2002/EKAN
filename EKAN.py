import torch
import torch.nn as nn
import numpy as np
from EKANLayer import *
from LBFGS import *
import os
import glob
from tqdm import tqdm
import random
from adan import Adan

class EKAN(nn.Module):
    def __init__(self, rep_in, rep_out, group, width=None, grid=3, k=3, base_fun=torch.nn.SiLU(), grid_eps=1.0, grid_range=[-1, 1], device='cpu', seed=0, classify=False):
        super(EKAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        middle_layers = [(c(group) if isinstance(c, Rep) else uniform_rep(c, group)) for c in width]
        reps = [self.rep_in] + middle_layers + [self.rep_out]

        self.act_fun = []
        self.depth = len(width) + 1

        for l in range(self.depth):
            sp_batch = EKANLayer(rep_in=reps[l], rep_out=reps[l + 1], num=grid, k=k, base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, device=device)
            self.act_fun.append(sp_batch)
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun
        
        self.device = device
        self.classify = classify

        in_dim = self.rep_in.size()
        gated_rep_in = gated(self.rep_in)
        gated_in_dim = gated_rep_in.size()
        self.lift = nn.Linear(in_dim, gated_in_dim, device=device)
        nn.init.orthogonal_(self.lift.weight)
        rep_W0 = gated_rep_in * self.rep_in.T
        Pw0_terms, multiplicities, perm, invperm = rep_W0.equivariant_projector(lazy=True)
        Pw0_terms = [torch.tensor(Pw0_term, dtype=torch.float32, device=device) for Pw0_term in Pw0_terms]
        self.Pw0 = lambda w: lazy_P(Pw0_terms, multiplicities, perm, invperm, w.unsqueeze(1))
        self.Pb0 = torch.tensor(gated_rep_in.equivariant_projector(), dtype=torch.float32, device=device)

        Wdim, bi_weight_proj_0 = bilinear_weights(gated_rep_in, gated_rep_in)
        self.bi_weight_proj_0 = bi_weight_proj_0
        self.bi_w0 = nn.Parameter(torch.randn(Wdim, device=device))

    def update_grid_from_samples(self, x):
        for l in range(self.depth):
            self.forward(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])

    def forward(self, x):
        x = x.to(self.device)
        W0 = self.Pw0(self.lift.weight.reshape(-1)).reshape(self.lift.weight.shape)
        b0 = self.Pb0 @ self.lift.bias
        x = x @ W0.T + b0
        bi_W0 = self.bi_weight_proj_0(self.bi_w0, x, self.device)
        x = .1 * (bi_W0 @ x[..., None])[..., 0] + x
        self.acts = []
        self.acts.append(x)
        for l in range(self.depth):
            x = self.act_fun[l](x)
            self.acts.append(x)
        x = x[..., : self.rep_out.size()]
        if self.classify:
            x = torch.sigmoid(x)
        return x

    def train(self, dataset, opt="Adan", steps=100, log=1, lamb=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., stop_grid_update_step=50, batch=-1, metrics=None, sglr_avoid=False, device='cpu'):
        
        def reg():
            reg_ = 0.
            reg_ += torch.sum(torch.abs(self.Pw0(self.lift.weight.reshape(-1))))
            reg_ += torch.sum(torch.abs(self.Pb0 @ self.lift.bias))
            for i in range(self.depth):
                reg_ += torch.sum(torch.abs(self.act_fun[i].weight_proj()))
                reg_ += torch.sum(torch.abs(self.act_fun[i].Pb @ self.act_fun[i].linear.bias))
            return reg_

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

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == 'Adan':
            optimizer = Adan(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
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

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
            else:
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
            reg_ = reg()
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        for _ in pbar:

            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(dataset['train_input'][train_id].to(device))

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam" or opt == 'Adan':
                pred = self.forward(dataset['train_input'][train_id].to(device))
                if sglr_avoid == True:
                    id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                    train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
                else:
                    train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
                reg_ = reg()
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))

            if _ % log == 0:
                pbar.set_description("train loss: %.2e | test loss: %.2e | reg: %.2e " % (train_loss.cpu().detach().numpy(), test_loss.cpu().detach().numpy(), reg_.cpu().detach().numpy()))

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(train_loss.cpu().detach().numpy())
            results['test_loss'].append(test_loss.cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

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