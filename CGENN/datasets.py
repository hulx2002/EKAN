import torch
import numpy as np
import pandas as pd

# modified from https://github.com/Rose-STL-Lab/LieGAN/blob/master/dataset.py
def TopTagging(path='../data/TopTagging/', n_component=3, N=-1, noise=0.0, device='cpu'):
    train_df = pd.read_hdf(path + 'train.h5', key='table').to_numpy()
    test_df = pd.read_hdf(path + 'test.h5', key='table').to_numpy()
    if N == -1 or N > train_df.shape[0]:
        N = train_df.shape[0]
    train_input = train_df[: N, : 4 * n_component]
    train_input = train_input * np.random.uniform(1 - noise, 1 + noise, size=train_input.shape)
    train_input = torch.FloatTensor(train_input)
    train_input = train_input.reshape(train_input.shape[0], -1, 4)
    test_input = test_df[:, : 4 * n_component]
    test_input = test_input * np.random.uniform(1 - noise, 1 + noise, size=test_input.shape)
    test_input = torch.FloatTensor(test_input)
    test_input = test_input.reshape(test_input.shape[0], -1, 4)
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