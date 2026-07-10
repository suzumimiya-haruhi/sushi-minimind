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


def get_lr():
    pass


def Logger():
    pass


def is_main_process():
    pass


def lm_checkpoint():
    pass


def init_distributed_mode():
    pass


def setup_seed():
    pass


def init_model():
    pass


class SkipBatchSampler(Sampler):
    pass