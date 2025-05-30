import torch
import pickle
import numpy as np
import pandas as pd

# modified from https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/datasets.py
def ParticleInteraction(x):
    P = x.reshape(-1, 4, 4)
    p1, p2, p3, p4 = P.permute(1, 0, 2)
    𝜂 = torch.diag(torch.tensor([1., -1., -1., -1.]))
    dot = lambda v1, v2: ((v1 @ 𝜂) * v2).sum(-1)
    Le = (p1[:, :, None] * p3[:, None, :] - (dot(p1, p3) - dot(p1, p1))[: , None, None] * 𝜂)
    L𝜇 = ((p2 @ 𝜂)[: , : , None] * (p4 @ 𝜂)[:, None, :] - (dot(p2, p4) - dot(p2, p2))[: , None, None] * 𝜂)
    M = 4 * (Le * L𝜇).sum(-1).sum(-1)
    y = M
    y = y[..., None]
    return y

# modified from https://github.com/Rose-STL-Lab/LieGAN/blob/master/dataset.py
def NBodyDataset(save_path='./data/NBody/2body-orbits-dataset.pkl', trj_timesteps=50, input_timesteps=4, output_timesteps=1, nbody=2, device='cpu'):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    train_data = data['coords']
    test_data = data['test_coords']
    feat_dim = nbody * 4
    if len(train_data.shape) == 2:
        train_data = train_data.reshape(-1, trj_timesteps, feat_dim)
    if len(test_data.shape) == 2:
        test_data = test_data.reshape(-1, trj_timesteps, feat_dim)
    if nbody == 2:
        train_data = train_data[:, :, [0, 2, 4, 6, 1, 3, 5, 7]]
        test_data = test_data[:, :, [0, 2, 4, 6, 1, 3, 5, 7]]
    elif nbody == 3:
        train_data = train_data[:, :, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]]
        test_data = test_data[:, :, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]]
    train_input, train_label, test_input, test_label = [], [], [], []
    train_N = train_data.shape[0]
    test_N = test_data.shape[0]
    trj_timesteps = train_data.shape[1]
    for i in range(train_N):
        for t in range(trj_timesteps - input_timesteps - output_timesteps):
            train_input.append(train_data[i, t : t + input_timesteps, :])
            train_label.append(train_data[i, t + input_timesteps : t + input_timesteps + output_timesteps, :])
    for i in range(test_N):
        for t in range(trj_timesteps - input_timesteps - output_timesteps):
            test_input.append(test_data[i, t : t + input_timesteps, :])
            test_label.append(test_data[i, t + input_timesteps : t + input_timesteps + output_timesteps, :])
    train_input = torch.tensor(np.array(train_input), dtype=torch.float32)
    train_label = torch.tensor(np.array(train_label), dtype=torch.float32)
    test_input = torch.tensor(np.array(test_input), dtype=torch.float32)
    test_label = torch.tensor(np.array(test_label), dtype=torch.float32)
    train_len = train_input.shape[0]
    test_len = test_input.shape[0]
    train_input = train_input.reshape(train_len, -1)
    train_label = train_label.reshape(train_len, -1)
    test_input = test_input.reshape(test_len, -1)
    test_label = test_label.reshape(test_len, -1)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)
    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)
    return dataset

# modified from https://github.com/Rose-STL-Lab/LieGAN/blob/master/dataset.py
def TopTagging(path='./data/TopTagging/', n_component=3, N=-1, noise=0.0, device='cpu'):
    train_df = pd.read_hdf(path + 'train.h5', key='table').to_numpy()
    test_df = pd.read_hdf(path + 'test.h5', key='table').to_numpy()
    if N == -1 or N > train_df.shape[0]:
        N = train_df.shape[0]
    train_input = train_df[: N, : 4 * n_component]
    train_input = train_input * np.random.uniform(1 - noise, 1 + noise, size=train_input.shape)
    train_input = torch.FloatTensor(train_input)
    test_input = test_df[:, : 4 * n_component]
    test_input = test_input * np.random.uniform(1 - noise, 1 + noise, size=test_input.shape)
    test_input = torch.FloatTensor(test_input)
    train_label = train_df[: N, -1]
    train_label = torch.LongTensor(train_label)
    test_label = test_df[:, -1]
    test_label = torch.LongTensor(test_label)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)
    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)
    return dataset