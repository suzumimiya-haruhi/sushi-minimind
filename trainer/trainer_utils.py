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

def get_model_params(model, config):
    """
    统计并打印模型参数量。
    total 表示模型全部参数量，active 表示 MoE 场景下每个 token 实际会激活的参数量。
    MoE 估算思路：base = total - 所有专家参数；active = base + 每个 token 选中的专家参数。
    打印中的 M 表示百万参数，A 表示 active params。
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')

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
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':   # 如果有权重，则加载权重
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device) # 加载权重
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer

class SkipBatchSampler(Sampler):    # 断点续训跳过前batch,用于传入dataloader
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):     # 生成器，装填batch
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []  # 清空要跳过的batch
                    continue
                yield batch     # 生成器返回这个batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch     # 尾部不丢弃

    def __len__(self):  # 返回剩余batch数量
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)