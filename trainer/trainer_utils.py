import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM


def get_lr(current_step, total_step, lr):   #余弦退火学习率
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step/total_step)))    # 1倍到0.1倍lr


def is_main_process():      # 判断是否是主进程
    return not dist.is_initialized() or dist.get_rank() == 0
    # (not dist.is_initialized()) or (dist.get_rank() == 0) 一边是true就为true

def Logger(content):
    if is_main_process():
        print(content)  # 只打印主进程


def lm_checkpoint():
    pass


def init_distributed_mode():    # 初始化分布式环境
    if int(os.environ.get('RANK', -1)) == -1:
        return 0

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):   # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False      # 减少卷积算法选择带来的随机性


# noinspection PyNoneFunctionAssignment
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model',
               save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)   # 分词器，分词规则按照tokenizer_path，最长匹配优先



class SkipBatchSampler(Sampler):
    pass