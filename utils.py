import numpy as np
import torch

# modified from https://github.com/KindXiaoming/pykan/blob/master/kan/utils.py
def create_dataset_randn(f, 
                   n_var=2,
                   range=1.,
                   train_num=1000, 
                   test_num=1000,
                   device='cpu',
                   seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_input = torch.randn(train_num, n_var) * range
    test_input = torch.randn(test_num, n_var) * range
           
    train_label = f(train_input)
    test_label = f(test_input)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset

from scipy.linalg import block_diag

def Kron(Ms):
    result = np.eye(1)
    for M in Ms:
        result = np.kron(result, M)
    return result

def Kronsum(Ms):
    result = np.zeros(1)
    for M in Ms:
        result = np.kron(result, np.eye(M.shape[0])) + np.kron(np.eye(result.shape[0]), M)
    return result

def DirectSum(Ms, multiplicities=None):
    multiplicities = [1 for M in Ms] if multiplicities is None else multiplicities
    Ms_all = [M for M, c in zip(Ms, multiplicities) for _ in range(c)]
    return block_diag(*Ms_all)

import hashlib

def consistent_hash(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest(), 16)

import matplotlib.pyplot as plt
import os

def visualization(results, name, folder='./results', classify=False):
    plt.plot(np.arange(len(results['train_loss'])), results['train_loss'], label='train')
    plt.plot(np.arange(len(results['test_loss'])), results['test_loss'], label='test')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('step')
    if classify:
        plt.ylabel('BCE')
    else:
        plt.ylabel('MSE')
    plt.tight_layout()

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(folder + '/' + name + '.png')