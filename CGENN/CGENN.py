import torch
import torch.nn as nn
import numpy as np
import os
import glob
from tqdm import tqdm
import random
from adan import Adan
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.linear import MVLinear
from models.modules.mvsilu import MVSiLU
from models.modules.mvlayernorm import MVLayerNorm

# modified from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/o5_cgmlp.py
class CGENN(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        n_layers=2,
        normalization_init=0,
        device='cpu',
        seed=0,
        dataset='TopTagging'
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.algebra = algebra
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.normalization_init = normalization_init

        if dataset == 'TopTagging':
            gp = nn.Sequential(
                MVLinear(self.algebra, in_features, 4 * in_features, subspaces=False),
                SteerableGeometricProductLayer(self.algebra, 4 * in_features)
            )
            self.gp = gp.to(device)
            mlp = []
            in_features = 4 * in_features
            for i in range(n_layers - 1):
                mlp.append(
                    nn.Sequential(
                        nn.Linear(in_features, hidden_features),
                        nn.ReLU()
                    )
                )
                in_features = hidden_features
            mlp.append(
                nn.Linear(in_features, out_features)
            )
            self.mlp = nn.Sequential(*mlp).to(device)

        self.device = device
        self.dataset = dataset

    def forward(self, x):
        x = x.to(self.device)
        x = self.algebra.embed_grade(x, 1)
        if self.dataset == 'TopTagging':
            x = self.gp(x)[..., 0]
            for layer in self.mlp:
                x = layer(x)
        if self.dataset == 'TopTagging':
            return torch.sigmoid(x)
        else:
            return x

    def train(self, dataset, opt="Adan", steps=100, log=1, loss_fn=None, lr=1., batch=-1, metrics=None, sglr_avoid=False, device='cpu'):
        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            if self.dataset == 'TopTagging':
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
