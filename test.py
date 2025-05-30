import argparse
from datasets import *
from utils import create_dataset_randn
from EKAN import *
from EMLP import *
from KAN import *
from MLP import *

def main(args):
    if args.dataset == 'ParticleInteraction':
        if args.group == 'SO13p':
            G = SO13p()
        elif args.group == 'SO13':
            G = SO13()
        else:
            G = Lorentz()
        rep_in = 4 * Vector
        rep_out = Scalar
        dataset = create_dataset_randn(f=ParticleInteraction, n_var=16, range=0.25, train_num=args.N, test_num=args.N, device=args.device, seed=args.seed)
        classify = False
    elif args.dataset == '3Body':
        if args.group == 'SO2':
            G = SO(2)
        else:
            G = O(2)
        rep_in = 24 * Vector
        rep_out = 6 * Vector
        dataset = NBodyDataset(input_timesteps=4, output_timesteps=1, save_path=f'./data/NBody/3body-orbits-dataset.pkl', nbody=3, device=args.device)
        classify = False
    elif args.dataset == 'TopTagging':
        if args.group == 'SO13p':
            G = SO13p()
        elif args.group == 'SO13':
            G = SO13()
        else:
            G = Lorentz()
        rep_in = args.n_component * Vector
        rep_out = Scalar
        dataset = TopTagging(n_component=args.n_component, N=args.N, noise=0.0, device=args.device)
        classify = True

    if args.network == 'EKAN':
        model = EKAN(rep_in=rep_in, rep_out=rep_out, group=G, width=args.width, grid=args.grid, device=args.device, seed=args.seed, classify=classify)
    elif args.network == 'EMLP':
        model = EMLP(rep_in=rep_in, rep_out=rep_out, group=G, width=args.width, device=args.device, seed=args.seed, classify=classify)
    elif args.network == 'KAN':
        model = KAN(rep_in=rep_in, rep_out=rep_out, group=G, width=args.width, grid=args.grid, device=args.device, seed=args.seed, classify=classify, augmentation=args.augmentation)
    elif args.network == 'MLP':
        model = MLP(rep_in=rep_in, rep_out=rep_out, group=G, width=args.width, device=args.device, seed=args.seed, classify=classify, augmentation=args.augmentation)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of parameters: {trainable_params}')

    if args.dataset == 'TopTagging':
        folder = './model_ckpt/' + args.dataset + str(args.n_component) + '/' + args.network
    else:
        folder = './model_ckpt/' + args.dataset + '/' + args.network

    if args.dataset in ['ParticleInteraction', 'TopTagging']:
        if args.augmentation:
            name = 'aug_' + str(args.N) + '_' + str(args.seed)
        elif args.network in ['KAN', 'MLP']:
            name = str(args.N) + '_' + str(args.seed)
        elif args.network in ['EKAN', 'EMLP']:
            name = str(G) + '_' + str(args.N) + '_' + str(args.seed)
    else:
        if args.augmentation:
            name = 'aug_' + str(trainable_params) + '_' + str(args.seed)
        elif args.network in ['KAN', 'MLP']:
            name = str(trainable_params) + '_' + str(args.seed)
        elif args.network in ['EKAN', 'EMLP']:
            name = str(G) + '_' + str(trainable_params) + '_' + str(args.seed)
    model.load_ckpt(name=name, folder=folder)

    if args.network in ['EKAN', 'KAN']:
        results = model.train(dataset=dataset, update_grid=False, steps=0, batch=args.batch, device=args.device)
    elif args.network in ['EMLP', 'MLP']:
        results = model.train(dataset=dataset, steps=0, batch=args.batch, device=args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ParticleInteraction')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--network', type=str, default='EKAN')
    parser.add_argument('--width', type=int, nargs='+', default=[])
    parser.add_argument('--grid', type=int, default=3)
    parser.add_argument('--batch', type=int, default=-1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--group', type=str, default='none')
    parser.add_argument('--n_component', type=int, default=3)
 
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    main(args)