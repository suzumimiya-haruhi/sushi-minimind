import math

import torch
from transformers import PretrainedConfig
from torch import nn


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind模型配置类
    """
    model_type = 'minimind'

    def __init__(self):
        super().__init__()
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None  # YaRN缩放参数，在后训练、微调、推理时都应传入


class RMSNorm(torch.nn.Module):
    """
    RMSNorm归一化，均方根缩放，乘可学习参数，不减去均值
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 避免除以0
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习参数

    def norm(self, x):
        """
        标准差
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播
        """
        return self.weight * self.norm(x.float()).type_as(x)


def precompute_freqs_cid(dim: int, end: int = int(32 * 1024), rope_base: float = 10000.0, rope_scaling: dict = None):
    """
    rope_scaling只在推理时传入进行YaRN，训练时不做缩放
    """
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))  # [:dim//2]是防御超过索引
    # 最终 freqs 的形状就是 [dim//2]，是传入的一个头，维度的一半
    attn_factor = 1.0  # 温度缩放系数
    # 预训练时attn_factor必须是1，推理和微调时attn_factor由rope_scaling转入的字典中的自断定义
    # 推理时rope_scaling缩放
    if rope_scaling is not None:  # 仅推理时传入
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            # orig_max: 预训练时的最大长度
            rope_scaling.get("factor", 16),
            # factor: 推理文本比训练最大长度的倍数，推理文本的长度上限的倍数，硬编为16
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )  # .get(), 如果字典中没有则用default，获得YaRN参数
        if end / orig_max > 1.0:  # 如果推理时传入的句子长度大于预训练时的长度，则做YaRN缩放
            """
            1、beta_fast=32: 作者认为（经验）训练时能在最大长度2048下转32圈的维度属于fast维度，也就是64个词就能转1圈的维度
            2、beta_slow=1: 2048个词才能刚好转一圈的维度，大于这个维度的维度称为slow维度，它们在训练时不能铺满整个面
            3、对于有fast维度，缩放为0；对于slow维度，缩放为1；中间的维度平滑缩放
            4、slow维度把整个圆 刚好 缩到扇形之内
            """
            # 计算波长对应的维度索引
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 计算fast维度和slow维度的索引
            low = max(math.floor(inv_dim(beta_slow)), 0)  # 索引值小于low的维度为fast
            high = min(math.ceil(inv_dim(beta_fast)), dim // 2 - 1)  # 索引值大于high的维度为fast
            # 计算每个区域的缩放系数
            # torch.clamp(input, min=None, max=None, *, out=None)，把input截断
            # torch.arange(dim//2)为索引,可用max(high - low , 0.001)防止除以0
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / (high - low), 0,1)
            # 计算缩放后的频率
            # factor: 推理文本比训练最大长度的倍数，推理文本的长度上限的倍数，硬编为16
            freqs = freqs * ((1 - ramp) + ramp /factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # 把token位置乘到频率上
    # 词向量方向分组旋转是前一半和后一半的对应位置分组旋转
    # [a,b,c,d,e,f]分为[a,d]\[b,e]\[c,f],旋转，所以freqs还原成dim维度直接在dim=-1拼接
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # 乘注意力分数
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
