import numpy as np
import matplotlib.pyplot as plt
import os

def visualization(results, name, folder='./results', dataset='TopTagging'):
    plt.plot(np.arange(len(results['train_loss'])), results['train_loss'], label='train')
    plt.plot(np.arange(len(results['test_loss'])), results['test_loss'], label='test')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('step')
    if dataset == 'TopTagging':
        plt.ylabel('BCE')
    else:
        plt.ylabel('MSE')
    plt.tight_layout()

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(folder + '/' + name + '.png')