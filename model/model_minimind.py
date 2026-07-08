import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型的配置类，用来保存网络结构、词表、位置编码、注意力头数、MoE 等超参数。
    继承 Hugging Face 的 PretrainedConfig 后，模型可以使用 from_pretrained/save_pretrained 等方式加载和保存配置。
    """
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        """
        初始化 MiniMind 的所有超参数。
        关键维度关系：head_dim = hidden_size / num_attention_heads。
        默认前馈网络中间层大小近似为 ceil(hidden_size * pi / 64) * 64，用 64 对齐方便计算。
        当 inference_rope_scaling=True 时启用 YaRN 风格的 RoPE 扩展配置。
        MoE 相关参数只在 use_moe=True 时被后续网络实际使用。
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None  # YaRN缩放参数，在后训练、微调、推理时都应传入
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(torch.nn.Module):
    """
    RMSNorm归一化，均方根缩放，乘可学习参数，不用方差，不减去均值
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 避免除以0
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习参数

    def norm(self, x):
        """
        标准差
        mean：对最后一维求均方，会降维成一个标量，所以用keepdim=True广播回去
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
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / (high - low), 0, 1)
            # 计算缩放后的频率
            # factor: 推理文本比训练最大长度的倍数，推理文本的长度上限的倍数，硬编为16
            freqs = freqs * ((1 - ramp) + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # 把token位置乘到频率上
    # 词向量方向分组旋转是前一半和后一半的对应位置分组旋转
    # [a,b,c,d,e,f]分为[a,d]\[b,e]\[c,f],旋转，所以freqs还原成dim维度直接在dim=-1拼接
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # 乘注意力分数
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    # return的结果应该是处理单个头的二维张量[seq_len,head_dim]
    return freqs_cos, freqs_sin # 返回一个处理单头的余弦正弦表


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    1、把位置编码应用在qk上，处理所有头
    2、二维平面RoPE([x1,x2])=[x1,x2]*cos + [-x2,x1]*sin
        q和k的维度是[batch_size,num_heads,seq_len,head_dim]:分别是批次方向，头数，token方向，每个头的隐藏层维度（是词向量方向）
        这里需要处理的是H维度
    3、传入的cos、sin应该有三个维度[batch_size,seq_len,head_dim]
        所以需要在传入前做cos = cos[None, :, :]   # 等价 cos.unsqueeze(0)
    4、... 代表一连串的 :   仅处理H时写成[...,H]
    5、* 是哈达玛积，相同位置元素相乘，@是矩阵乘法
    """

    def rotate_half(x):
        """
        传入q或者k，把H维度从[x1,x2]转换成[-x2,x1]
        [a,b,c,d,e,f] -> [-d,-e,-f,a,b,c]
        """
        return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
        # torch.cat的序列参数（第一个参数）可以用元组或列表

    cos = cos.unsqueeze(unsqueeze_dim)  # 在维度1的位置插入一个维度，把[L,D]变成[L,1,D]
    sin = sin.unsqueeze(unsqueeze_dim)  # 此时的cos和sin的维度是[L,1,D],qk的维度是[B,L,H,D]
    """
    哈达玛积广播：
    只有要广播的维度有一方维度数是1才能广播，其他情况都不行
    张量 1：[B, 1, L, D]，张量 2：[1, H, L, D] 可以广播
    张量 1：[1, 2, L, D]，张量 2：[B, H, L, D] 不能广播
    """
    q_embed = ((q * cos) + (rotate_half(q) * sin)).to(q.dtype)
    k_embed = ((k * cos) + (rotate_half(k) * sin)).to(k.dtype)
    """
                        形状的最后一块拼图   
    四维的qk乘三维的cos,torch自动从-1维度对齐补全cos维度,并把值为1的维度广播
    """
    # tensor.to(q.dtype) : 把张量数据类型转换成q中元素的数据类型
    # tensor.type_as(x) : 转换元素类型和所在设备，等价tensor.to(dtype=x.dtype, device=x.device)
    # np_arr.astype(np.float32) : numpy转换元素类型
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    多个q头对应一个kv对，适配分组多头注意力GQA
    所以把kv先复制成和q头一致
    输入形状:[batch，seq_len,num_key_value_heads,head_dim]。
    输出形状:[batch，seq_len, num_key_value_heads * n_rep，head_dim]。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        # 如果n_rep=1，则不需要复制，直接返回x
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen,
                                                                                              num_key_value_heads * n_rep,
                                                                                              head_dim)
    # x[:, :, :, None, :]等价于x.unsqueeze(3),在第三维度增加一个维度，维度数为1
    # .expand()以广播的形式把维度数为1的维度重复成n_rep，不复制节省内存


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None \
            else config.num_key_value_heads  # 如果config中有kv头数就用它，不然kv头数等于总注意力头数
        self.n_local_heads = config.num_attention_heads  # 放在本地显存的头数
        self.n_local_kv_heads = config.num_key_value_heads  # 放在本地显存的kv头
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 一个kv对应几个q
        self.head_dim = config.head_dim  # 注意力头维度
        self.is_causal = True  # 使用因果注意力掩码

        # 注意力权重矩阵QKV，一般Q是方阵 即heads_size = num_kv_value_heads * head_dim
        # WK、WV和WQ的输出差一个n_rep的倍数
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # O矩阵，输入多头拼接结果，融合多头的特征，输出矩阵做残差连接，一般是方阵
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # 在算注意力分数前对q和k做标准化
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)  # 残差dropout
        self.dropout = config.dropout

        # 闪存式高效注意力，硬件优化
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        # 算权重矩阵------此时qkv形状：[batch_size, seq_len, dim]
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        # 拆分为多头------此时q形状：[batch_size, seq_len, num_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # 对qk标准化------形状不变
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        # 位置编码------形状不变
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 如果有kv缓存,就把缓存和当前的kv拼接
        if past_key_value:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 在seq_len维度上拼接
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 保存缓存
        past_kv = (xk, xv) if use_cache else None

        """
        qk计算位置编码
        kv做缓存，缓存已经完成位置编码的k
        """

        # 交换seq和head的维度，并把kv和q的头数对齐
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 如果用闪存式高效注意力
        if self.flash and seq_len>1 and (not self.is_causal or past_key_value is None) \
            and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout
                                                    if self.training
                                                    else 0.0, is_causal=self.is_causal)
        else:
            # 手动实现q乘k转置除以维度的方根
            # [seq_q, dim] @ [dim, seq_k] = [seq_q, seq_k]
            # scores.shape: [batch_size, num_heads, seq_q, seq_k]
            scores = (xq @ xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            # todo1  懂了
            if self.is_causal:
                # torch.full(...,-inf).triu: 上三角矩阵inf掩码，triu：上三角
                # 把seq_k维度上的未来token加掩码，只对最后的seq_len 个token进行填充，刚好是一个上三角
                scores[:,:,:,-seq_len:] += torch.full((seq_len, seq_len), float('-inf'), device=scores.device).triu(1)
            if attention_mask :
                # 掩码pading的无效token
                # attention_mask形状：[B, L] 在1，2维度升维 --> [B, 1, 1, L]
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # 多头合并
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)    # 融合
        output = self.resid_dropout(output)     # dropout一下，准备残差
        return output, past_kv

class FeedForward(nn.Module):
    """
    silu激活门控的前馈神经网络
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)   # 走激活
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * self.act_fn(self.gate(x)))


# todo1 MOEFFN
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False) # 计算token走哪个专家
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        # 一个专家是一个完整的FFN
        self.act_fn = ACT2FN[config.hidden_act] # 自动选择激活函数

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)     # 把batch和seq len压成一维，适配topk，x_flat.shape: [batch_size*seq_len, hidden_dim]
        scores = F.softmax(self.gate(x_flat), dim=-1)    # 对每个token走哪个专家打分，形状：哪句的哪个token，对应的八个专家的概率
        topk_weights, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        # num_experts_per_tok：每个token走几个专家，topk_weights：专家权重，topk_indices：专家id
        # topk_weights, topk_indices 形状：[batch_size*seq_len, num_experts_per_tok]
        if self.config.norm_topk_prob:  # 专家权重的归一化
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)    # return目标
        for i, expert in enumerate(self.experts):   # 遍历每个专家，筛选需要经过这个专家的token
            mask = (topk_idx == i) # 逐元素比较，筛选出哪些token走当前循环的专家
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # any(dim)压缩dim维度成bool
                # nonzero()取True的下标，把稀疏变稠密
                # flatten()直接拉平成一维，
                weight = topk_weights[mask].view(-1, 1)
                y.index_add_(0, token_idx, weight * expert(x_flat[mask]))
            elif self.training:
                # 如果在训练且没有token走这个专家，但是专家需要梯度更新，通过这一行把闲置专家拉进计算图
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

            if self.train and self.config.router_aux_loss_coef > 0: # 调整专家loss
                load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
                # one hot:把稠密的专家id加一个维度变成独热，比如[0,2]->[[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0]]
                # mean(0)求每个位置、每个专家的概率
                self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
                # 通过广播把二维的load和一维的scores.mean(0)逐元素相乘，然后逐元素相加，再乘系数
            else:
                self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)


class MiniMindBlock(nn.Module):
    """
    一个decoder块，结构是:
    h = x + Attention(RMSNorm(x));
    out = h + MLP(RMSNorm(x)).
    MLP可以是FFN或者MOEFFN
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):  # layer_id: 当前层id
        super().__init__()
        self.self_attn = Attention(config)  # 自注意力层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)     # 输入层归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)    # 注意力后层归一化
        self.mlp = FeedForward(config) # if config.use_moe else MOEFeedForward(config)

    def forward(self,hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        :param hidden_states: 隐藏状态
        :param position_embeddings: 位置编码器
        :param past_key_value: kv缓存
        :param use_cache: 是否用kv缓存
        :param attention_mask: 注意力掩码
        """
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )   # 注意力层输出隐藏状态和kv缓存
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value # 返回隐藏状态和kv缓存


# noinspection PyAttributeOutsideInit
class MiniMindModel(nn.Module):
    """
    模型主干
    输入词id输出隐藏层特征
    input：ids -> 词嵌入向量 -> transformer层 * n -> output：RMSNorm
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size     # 词表大小
        self.num_hidden_layers = config.num_hidden_layers   # transformer层数
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 词嵌入向量 把词表映射成隐藏维度大小
        # 定义Embedding类时，就创建了一张vocab_size * hidden_size的词向量表，跟随训练更新
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(config.num_hidden_layers)])
        # 把num_hidden_layers个transformer层堆叠起来
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)    # transformer循环后的层归一化
        freqs_cos, freqs_sin = precompute_freqs_cid(dim=config.head_dim,    # 一个头的维度
                                                    end=config.max_position_embeddings,     # 最大位置编码
                                                    rope_base=config.rope_theta,    # rope的底数1000000
                                                    rope_scaling=config.rope_scaling)    # rope的缩放
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 把cos注册到模型跟随模型设备，但是没有梯度不参与更新
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)  # persistent=False 不存进权重

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape    # 传入批次和词id矩阵
        if hasattr(past_key_values, 'layers'):
            """
            hasattr:如果past_key_values是有带有layers属性的对象则为true
            训练的时候没有past_key_values，所以past_key_values总为none，这里做一个防御
            推理的时候逐token生成，会生成一个带有layer属性的DynamicCache类，所以只有训练的时候有缓存
            """
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)  # 继承或初始化缓存，None是不可变对象
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        """past_key_values是保存每层kv对元组的元组，结构是(((k1),(v1)),((k2),(v2)),...)
        past_key_values[0] = ((k1),(v1))  第一层的kv对
        past_key_values[0][0] = (k1), k.shape = (bsz, seq_len, num_heads, head_dim)
        seq_len是缓存的句子token的长度"""
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # 在词表中根据ids取向量
        if self.freqs_cos[0, 0] == 0:
            """
            self.register_buffer("freqs_sin", freqs_sin, persistent=False),中persistent=False，正余弦表不保存
            当模型后训练、微调等时，freqs_cos, freqs_sin会被置为0，所以重新计算正余弦表
            """
            freqs_cos, freqs_sin = precompute_freqs_cid(dim=self.config.head_dim,
                                                        end=self.config.max_position_embeddings,
                                                        rope_base=self.config.rope_theta,
                                                        rope_scaling=self.config.rope_scaling)
            self.freqs_cos = freqs_cos.to(hidden_states.device) # self.freqs_cos、sin已经在init中注册过，这里直接赋值
            self.freqs_sin = freqs_sin.to(hidden_states.device)
        postion_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length],
                              self.freqs_sin[start_pos:start_pos + seq_length]) # 只给新传入的token的位置编码器
        persents = []   # 当前kv缓存
        for layer, past_key_value in zip(self.layers, past_key_values): #循环多次transformer
            hidden_states, present = layer(hidden_states,
                                           postion_embeddings,
                                           past_key_value=past_key_value,
                                           use_cache=use_cache,
                                           attention_mask=attention_mask)
            persents.append(present)
        hidden_states = self.RMSNorm(hidden_states)
        aux_loss = None # MOE损失
        return hidden_states, persents, aux_loss


class MiniMindForCausalLM(nn.Module):
    """
    主干后面的功能头
    训练：forward，生成：generate
    输入隐藏层特征输出词（算loss）
    input：rmsnorm -> 线性层 -> softmax -> 选词
    """
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}  # 共享嵌入层和输出层的词向量表
    def __init__(self, config: MiniMindConfig=None):
        self.config = config or MiniMindConfig()
        super().__init__()
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()    # 递归初始化所有网络参数

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, labels=None, logits_to_keep=0, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[..., slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), logits[..., 1:, :].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, aux_loss=aux_loss)