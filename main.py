import torch
from args import get_args
from utils import set_seed

from datasets.lm import PennTreebank, WikiText2


WANDB_STEP = -1


def main():
    global WANDB_STEP
    args = get_args()
    print(args)

    #utils里面的函数,设置random参数的种子
    set_seed(args.seed)      

    #设置使用gpu/cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    aux_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.aux_device = aux_device


    #选择dataset（在datasets/lm.py中）
    if args.dataset == "ptb":
        Dataset = PennTreebank
    elif args.dataset == "wikitext2":
        Dataset = WikiText2




