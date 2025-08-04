import torch
import random
import numpy as np





def set_seed(seed):
    #设置不同库的seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_name(config):
    return "_".join([
        config.dataset,
        config.iterator,
        config.model,
        f"k{config.num_classes}",
        f"wps{config.words_per_state}",
        f"spw{config.states_per_word}",
        f"tspw{config.train_spw}",
        #f"ff{config.ffnn}",
        f"ed{config.emb_dim}",
        f"d{config.hidden_dim}",
        f"cd{config.char_dim}",
        f"dp{config.dropout}",
        f"tdp{config.transition_dropout}",
        f"cdp{config.column_dropout}",
        f"sdp{config.start_dropout}",
        f"dt{config.dropout_type}",
        f"wd{config.word_dropout}",
        config.bsz_fn,
        f"b{config.bsz}",
        config.optimizer,
        f"lr{config.lr}",
        f"c{config.clip}",
        f"tw{config.tw}",
        f"nas{config.noise_anneal_steps}",
        f"pw{config.posterior_weight}",
        f"as{config.assignment}",
        f"nb{config.num_clusters}",
        f"nc{config.num_common}",
        f"ncs{config.num_common_states}",
        f"spc{config.states_per_common}",
        f"n{config.ngrams}",
        f"r{config.reset_eos}",
        f"ns{config.no_shuffle_train}",
        f"fc{config.flat_clusters}",
        f"e{config.emit}",
        f"ed{'-'.join(str(x) for x in config.emit_dims) if config.emit_dims is not None else 'none'}",
        f"nh{config.num_highway}",
        f"s{config.state}",
    ])