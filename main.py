import torch
import torchtext
import wandb
from args import get_args
from utils import set_seed, get_name

from datasets.lm import PennTreebank, WikiText2
from datasets.data import BucketIterator, BPTTIterator


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


    #选择dataset,设置train、valid、test（在datasets/lm.py中）
    TEXT = torchtext.data.Field(batch_first=True)
    if args.dataset == "ptb":
        Dataset = PennTreebank
    elif args.dataset == "wikitext2":
        Dataset = WikiText2
    train, valid, test = Dataset.splits(TEXT, newline_eos=True)

    #构建词汇表
    TEXT.build_vocab(train)
    V = TEXT.vocab

    #batch_size划分标准
    def batch_size_tokens(new, count, sofar):
        return max(len(new.text), sofar)
    def batch_size_sents(new, count, sofar):
        return count

    #选择迭代器 bucket按照句子/bptt按照token（在datasets/data.py中）
    if args.iterator == "bucket":
        train_iter, valid_iter, test_iter = BucketIterator.splits(
            (train, valid, test),
            batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
            device = device,
            sort_key = lambda x: len(x.text),
            batch_size_fn = batch_size_tokens if args.bsz_fn == "tokens" else batch_size_sents,)
    elif args.iterator == 'bptt':
        train_iter, valid_iter, test_iter = BPTTIterator.splits(
            (train, valid, test),
            batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
            device = device,
            bptt_len = args.bptt,
            sort = False,
        )
    else:
        raise ValueError(f"Invalid Iterator {args.iterator}")
    
    #是否打乱每个epoch的train数据
    if args.no_shuffle_train:
        train_iter.shuffle = False

    #初始化wandb
    name = get_name(args)
    import tempfile
    wandb.init(project="hmm-lm", name=name, config=args, dir=tempfile.mkdtemp())
    args.name = name

    #导入hmmlm模型
    model = None

    #仅评测模式
    if args.eval_only:
        pass


    #选择更新参数的optimizer和调整学习率的scheduler



    





