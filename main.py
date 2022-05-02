import argparse

import torch
import wandb
wandb.login()

from dataloader import get_dataloaders
from utils import get_model
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'svhn'])
parser.add_argument('--model', required=True, choices=['mixer', 'qmixer', 'squeeze_net', 'alex_net', 'resnet18', 'qresnet18', 'resnet34', 'qresnet34'])
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--eval-batch-size', type=int, default=1024)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--epochs', type=int, default=300)
# parser.add_argument('--precision', type=int, default=16)

parser.add_argument('--patch-size', type=int, default=4)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--hidden-c', type=int, default=512)
parser.add_argument('--hidden-s', type=int, default=64)
parser.add_argument('--num-layers', type=int, default=8)
parser.add_argument('--drop-p', type=int, default=0.)
parser.add_argument('--off-act', action='store_true', help='Disable activation function') # TODO: remove
parser.add_argument('--act', type=str, default="gelu", help='Choose activation function')
# parser.add_argument('--co', action='store_true', help='Use a copilot')
parser.add_argument('--is-cls-token', action='store_true', help='Introduce a class token.')
parser.add_argument('--groups', type=int, default=1)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--min-lr', type=float, default=1e-6)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine'])
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--weight-decay', type=float, default=5e-5)
parser.add_argument('--off-nesterov', action='store_true')
parser.add_argument('--label-smoothing', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--warmup-epoch', type=int, default=5)
parser.add_argument('--autoaugment', action='store_true')
parser.add_argument('--clip-grad', type=float, default=0, help="0 means disabling clip-grad")
parser.add_argument('--cutmix-beta', type=float, default=1.0)
parser.add_argument('--cutmix-prob', type=float, default=0.)

args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.nesterov = not args.off_nesterov
torch.random.manual_seed(args.seed)

experiment_name = f"{args.model}_{args.dataset}_{args.optimizer}_{args.scheduler}"

if args.autoaugment:
    experiment_name += "_aa"
if args.clip_grad:
    experiment_name += f"_cg{args.clip_grad}"
if args.cutmix_prob>0.:
    experiment_name += f'_cm'

if args.model == 'mixer':
    experiment_name += "_l" + str(args.num_layers)
    if args.is_cls_token:
        experiment_name += f"_cls"
elif args.model == 'qmixer':
    experiment_name += "_l" + str(args.num_layers)
    experiment_name += "_g" + str(args.groups)
    if args.is_cls_token:
        experiment_name += f"_cls"

elif args.model == 'squeeze_net':
    experiment_name += f"_{args.act}"
elif args.model == 'alex_net':
    experiment_name += f"_{args.act}"
elif args.model == 'resnet18':
    experiment_name += f"_{args.act}"
elif args.model == 'resnet34':
    experiment_name += f"_{args.act}"



if __name__=='__main__':
    with wandb.init(project=args.model, config=args, name=experiment_name):
        train_dl, test_dl = get_dataloaders(args)
        model = get_model(args)
        print(model)
        trainer = Trainer(model, args)
        trainer.fit(train_dl, test_dl)