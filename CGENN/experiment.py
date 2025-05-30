import argparse
from datasets import *
from utils import visualization
from algebra.cliffordalgebra import CliffordAlgebra
from CGENN import *

def main(args):
    if args.dataset == 'TopTagging':
        algebra = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
        in_features = args.n_component
        out_features = 1
        dataset = TopTagging(n_component=args.n_component, N=args.N, noise=0.0, device=args.device)

    model = CGENN(algebra=algebra, in_features=in_features, hidden_features=args.hidden_features, out_features=out_features, n_layers=args.n_layers, device=args.device, seed=args.seed, dataset=args.dataset)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of parameters: {trainable_params}')

    results = model.train(dataset=dataset, opt=args.opt, steps=args.steps, log=1, lr=args.lr, batch=args.batch, device=args.device)

    if args.dataset == 'TopTagging':
        name = args.dataset + str(args.n_component) + '_' + str(args.N) + '_' + str(args.seed)
    model.save_ckpt(name=name)
    # visualization(results=results, name=name, dataset=args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TopTagging')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--hidden_features', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--opt', type=str, default='Adan')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1.)
    parser.add_argument('--batch', type=int, default=-1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_component', type=int, default=3)
    
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    main(args)