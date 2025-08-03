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