import argparse
from datasets import *
from utils import create_dataset_randn
from EKAN import *
from EMLP import *
from KAN import *
from MLP import *

def test_equi(model, dataset, G, rep_in, rep_out, batch=-1, device='cpu'):
    loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
    if batch == -1 or batch > dataset['test_input'].shape[0]:
            batch_size_test = dataset['test_input'].shape[0]
    else:
        batch_size_test = batch

    with torch.no_grad():
        overall_test_loss = 0.
        for i in range(0, dataset['test_input'].shape[0], batch_size_test):
            test_id = np.arange(i, min(i + batch_size_test, dataset['test_input'].shape[0]))
            g = G.sample()
            y1 = model(torch.einsum('ik,jk->ij', dataset['test_input'][test_id].to(device), torch.from_numpy(rep_in.rho(g)).to(torch.float32).to(device)))
            y2 = torch.einsum('ik,jk->ij', model(dataset['test_input'][test_id].to(device)), torch.from_numpy(rep_out.rho(g)).to(torch.float32).to(device))
            overall_test_loss += len(test_id) * loss_fn_eval(y1, y2)
        overall_test_loss /= dataset['test_input'].shape[0]
        print(f'overall test loss: {overall_test_loss}')

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
        rep_in = 3 * Vector
        rep_out = Scalar
        dataset = TopTagging(n_component=3, N=args.N, noise=0.0, device=args.device)
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

    test_equi(model=model, dataset=dataset, G=G, rep_in=rep_in(G), rep_out=rep_out(G), batch=args.batch, device=args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='O5Synthetic')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--network', type=str, default='EKAN')
    parser.add_argument('--width', type=int, nargs='+', default=[])
    parser.add_argument('--grid', type=int, default=3)
    parser.add_argument('--batch', type=int, default=-1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--group', type=str, default='none')
 
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    main(args)