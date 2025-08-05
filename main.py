import torch, torchtext, wandb, time, sys, math

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from args import get_args
from utils import set_seed, get_name

from datasets.lm import PennTreebank, WikiText2
from datasets.data import BucketIterator, BPTTIterator


WANDB_STEP = -1
valid_schedules = ["reducelronplateau"]

def update_best_valid():
    return

def report(losses, n, prefix, start_time=None):
    loss = losses.evidence
    elbo = losses.elbo
    str_list = [
        f"{prefix}: log_prob = {loss:.2f}",
        f"xent(word) = {-loss /n :.2f}",
        f"ppl = {math.exp(-loss / n):.2f}",
    ]
    if elbo is not None:
        pass

    return

def eval_loop():
    return

def cached_eval_loop():
    return

def mixed_cached_eval_loop():
    return

def train_loop():
    return



def count_params(model):
    return (
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

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
    from models.factoredhmmlm import FactoredHmmLm              #别忘去建立模型
    model = FactoredHmmLm(V, args)  #V是词表，args是config里的参数配置文件
    model.to(device)
    print(model)

    #hmmlm的参数
    num_params, num_trainable_params = count_params(model)
    print(f"Num params, trainable:{num_params:,}, {num_trainable_params:,}")
    wandb.run.summary["num_params"] = num_params


    #训练后仅评测模式
    if args.eval_only:
        #载入训练好的params
        model.load_state_dict(torch.load(args.eval_only)["model"])  

        #返回valid的时间戳
        v_start_time = time.time()      
        #选用不同的loop
        if args.model == "mshmm" or args.model == "factoredhmm":
            if args.num_classes > 2 ** 15 :
                eval_fn = mixed_cached_eval_loop
            else:
                eval_fn = cached_eval_loop
        elif args.model == "hmm":
            eval_fn = cached_eval_loop
        else:
            eval_fn = eval_loop
        #返回valid的结果
        valid_losses, valid_n =  eval_fn(args, V, valid_iter, model)
        report(valid_losses, valid_n, f"Valid perf", v_start_time)

        #返回test的时间戳
        t_start_time = time.time()
        #返回test的结果
        test_losses, test_n = eval_fn(args, V, test_iter, model)
        report(test_losses, test_n, f"Test perf", t_start_time)

        sys.exit()

    #选择更新参数的optimizer和调整学习率的scheduler
    parameters = list(model.parameters())
    if args.optimizer == "adamw":
        optimizer = AdamW(
            parameters,
            lr = args.lr, 
            betas = (args.beta1, args.beta2),
            weight_decay = args.wd,
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            parameters,
            lr = args.lr,
        )
    if args.schedule == "reducelronplateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor = 1. /args.decay,
            patience = args.patience,
            verbose = True,
            mode = "max",
        )
    elif args.schedule == "noam":
        warmup_steps = args.warmup_steps
        def get_lr(step):
            scale = warmup_steps ** 0.5 * min(step ** (-0.5), step * warmup_steps ** (-1.5))
            return args.lr * scale
        scheduler = LambdaLR(
            optimizer,
            get_lr,
            last_epoch=-1,
            verbose = True
        )
    else:
        raise ValueError("Invalid schedule options")
    
    #训练过程
    for e in range(args.num_epochs):
        start_time = time.time()

        #每轮epoch清零数据
        if args.log_counts > 0 and args.keep_counts > 0 :
            model.state_counts.fill_(0)
        
        #训练流程
        train_losses, train_n = train_loop(
            args, V, train_iter, model,
            parameters, optimizer, scheduler,
            valid_iter = valid_iter if not args.overfit else None,
            verbose = True
        )
        total_time = report(train_losses, train_n, f"Train epoch {e}", start_time)

        #评测流程，仅valid
        v_start_time = time.time()
        if args.model == "mshmm" or args.model == "factoredhmm":
            if args.num_classes > 2 ** 15:
                eval_fn = mixed_cached_eval_loop
            else:
                eval_fn = cached_eval_loop
        elif args.model == "hmm":
            eval_fn = cached_eval_loop
        else:
            eval_fn = eval_loop
        valid_losses, valid_n  = eval_fn(args, V, valid_iter, model)
        report(valid_losses, valid_n, f"Valid epoch {e}", v_start_time)

        if args.schedule in valid_schedules:
            scheduler.step(
                valid_losses.evidence if not args.overfit else train_losses.evidence)       #overfit模式是只用一小个batch去重复训练，来测试
        




    





