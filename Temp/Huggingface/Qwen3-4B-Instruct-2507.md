---
date: 2025-10-12
author:
  - Siyuan Liu
tags:
  - Huggingface
---
# 作者简述
本篇文章主要用于记录Qwen3-4B-Instruct-2507模型的源码理解，用关键module的相对路径作为标题

---
# .\transformers\models\qwen3\configuration_qwen3.py
## base_model_tp_plan

```
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
```
- - `layers`: 指的是模型的主体部分，即 Transformer 层的列表  
- `*`: 通配符，表示**任意一层**（例如第 0 层、第 1 层...直到最后一层）
- `self_attn`: 指的是该层内的自注意力（Self-Attention）模块
- `q_proj`: 指的是自注意力模块中用于生成 Query（查询）的**投影层 (projection layer)** 的权重
- **`"colwise"` (Column-wise / 按列切分)**
    - 将权重矩阵沿着**列（column）** 的方向进行切分。
    - **数学原理**: 对于矩阵乘法 `Y = XA`，如果我们将矩阵 `A` 按列切分为 `[A_1, A_2, ..., A_n]`，那么输出 `Y` 也可以被自然地切分为 `[XA_1, XA_2, ..., XA_n]`。每个 GPU 计算自己分到的那部分 `XA_i`，得到输出的一部分 `Y_i`。这个过程几乎不需要通信。
- **`"rowwise"` (Row-wise / 按行切分)**
    - 将权重矩阵沿着**行（row）** 的方向进行切分。
    - **数学原理**: 这种方式通常用于一个计算模块的**输出投影层**。它的输入 `X` 通常是前一个 `colwise` 层并行计算得到的、已经被切分的结果。当它与一个按行切分的权重 `A` 相乘后，每个 GPU 会得到一个部分和（partial sum）。最后需要一个**全局规约（All-Reduce）**操作，将所有 GPU 上的部分和加起来，得到最终的、完整的输出结果。这个 `All-Reduce` 操作是必需的通信步骤。

所以，`layers.*.self_attn.q_proj`键的含义是：“模型中**每一层**的**自注意力模块**里的**q_proj权重**”

---
## base_model_pp_plan
```
base_model_pp_plan = {
    "embed_tokens": (["input_ids"], ["inputs_embeds"]),
    "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
    "norm": (["hidden_states"], ["hidden_states"]),
}
```
**字典的键代表了模型中可以被视为一个独立流水线阶段的宏观功能模块:**
- `"embed_tokens"`: 模型的词嵌入层。它将输入的 token ID 转换为向量表示。这通常是流水线的第一个操作
- `"layers"`: 模型的主体，即所有 Transformer 层的堆栈。这个部分是计算量最大的，也是流水线并行切分的主要对象
- `"norm"`: 在所有 Transformer 层之后，但在最终输出头之前的最终归一化层（Final LayerNorm）

**字典的值是一个元组 `(inputs, outputs)`，其中 `inputs` 和 `outputs` 都是列表，定义了该模块的输入和输出张量的名称:**
- `"embed_tokens": (["input_ids"], ["inputs_embeds"])`
    - **输入**: 需要一个名为 `"input_ids"` 的张量（即 tokenized 后的文本）
    - **输出**: 会产生一个名为 `"inputs_embeds"` 的张量（即词嵌入向量）
    - **含义**: 当 `embed_tokens` 模块被放置在流水线的第一个阶段（例如 GPU 0）时，框架知道需要将原始数据中的 `input_ids` 提供给它，并将其输出 `inputs_embeds` 传递给下一个阶段。
- `"layers": (["hidden_states", "attention_mask"], ["hidden_states"])`
    - **输入**: 需要两个张量，一个名为 `"hidden_states"`（上一层的输出，对于第一层来说就是 `inputs_embeds`），另一个名为 `"attention_mask"`（用于屏蔽 padding tokens）
    - **输出**: 会产生一个更新后的 `"hidden_states"` 张量
    - **含义**: 当 `layers` 模块被切分到多个 GPU 上时（例如，GPU 0 负责 0-7 层，GPU 1 负责 8-15 层），框架知道在 GPU 0 和 GPU 1 之间需要传递更新后的 `hidden_states`。同时，它也知道 `attention_mask` 这个张量需要从头到尾贯穿整个流水线，并提供给每一层的计算
- `"norm": (["hidden_states"], ["hidden_states"])`
    - **输入**: 需要最后一个 Transformer 层输出的 `"hidden_states"`
    - **输出**: 产生归一化后的 `"hidden_states"`
    - **含义**: 当 `norm` 模块被放置在流水线的最后一个阶段时，框架知道应该从前一个阶段接收 `hidden_states`，并将其计算结果作为整个流水线（除输出头外）的最终结果

---
# .\transformers\models\qwen2\tokenization_qwen2_fast.py
底层逻辑使用的`GPT2TokenizerFast`

快速分词器和慢速分词器的区别是：快速分词器底层实现是Rust写的，而慢速分词器是纯Python

---
# .\transformers\models\qwen3\modular_qwen3.py
``` 
class Qwen3RMSNorm(Qwen2RMSNorm):  
    pass
```
- Qwen3 的 RMSNorm（均方根层归一化） 与 Qwen2 完全相同

```
class Qwen3MLP(GemmaMLP):  
    pass
```
- Qwen3 的 MLP（前馈网络） 采用了和 Gemma 模型完全相同的实现

```  
class Qwen3Attention(LlamaAttention):  
    def __init__(self, config: Qwen3Config, layer_idx: int):  
        super().__init__(config, layer_idx)  
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!  
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape  
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None  
```
- 继承自LlamaAttention, 这表明 Qwen3 的注意力机制是基于 Llama 的，但内部有自定义的修改
- `self.q_norm` `self.k_norm`: 是 Qwen3 注意力机制的核心改动。它为Q和K分别增加了独立的 **RMSNorm** 层
- self.sliding_window: 根据配置判断当前层是否使用**滑动窗口注意力（Sliding Window Attention）**

```
    def forward(  
        self,  
        hidden_states: torch.Tensor,  
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  
        attention_mask: Optional[torch.Tensor],  
        past_key_values: Optional[Cache] = None,  
        cache_position: Optional[torch.LongTensor] = None,  
        **kwargs: Unpack[FlashAttentionKwargs],  
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:  
        input_shape = hidden_states.shape[:-1]  
        hidden_shape = (*input_shape, -1, self.head_dim)  
  
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  
```
- 计算 **Query, Key, Value**。
	- `self.q_proj(hidden_states)`: 对输入 `hidden_states` 进行线性变换得到 Q
	- `.view(hidden_shape)`: 将 Q 重塑为 `(batch_size, seq_len, num_heads, head_dim)` 的形状
	- `self.q_norm(...)`: **在重塑后、转置前**，对 Q 应用 `RMSNorm` 进行归一化。`k_norm` 对 K 的处理同理
	- `value_states`: Value (V) 的计算与 Llama 相同，**没有**应用归一化
	- `.transpose(1, 2)`: 交换维度以适应多头注意力的计算格式

```
        cos, sin = position_embeddings  
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```
- 将**旋转位置编码（RoPE）** 应用于 Q 和 K，以注入位置信息

```
if past_key_values is not None:  
            # sin and cos are specific to RoPE models; cache_position needed for the static cache  
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)  
```
- 处理**键值缓存（KV Cache）**。在生成文本时，为了避免重复计算，会将之前时间步的 K 和 V 缓存起来。`past_key_values.update` 方法负责更新和返回完整的 K 和 V 序列

```
attention_interface: Callable = eager_attention_forward  
        if self.config._attn_implementation != "eager":  
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  
```
- 根据配置选择具体的注意力计算函数。可以是默认的 PyTorch 实现 (`eager_attention_forward`)，也可以是更高效的实现，如 **Flash Attention** 或 **SDPA**

```
attn_output, attn_weights = attention_interface(  
            self,  
            query_states,  
            key_states,  
            value_states,  
            attention_mask,  
            dropout=0.0 if not self.training else self.attention_dropout,  
            scaling=self.scaling,  
            sliding_window=self.sliding_window,  # diff with Llama  
            **kwargs,  
        )  
```
- 调用选择的注意力函数进行计算
- `sliding_window=self.sliding_window`: 将前面设置的**滑动窗口大小**传递给注意力函数，这是与 Llama 的一个显著不同点

```
attn_output = attn_output.reshape(*input_shape, -1).contiguous()  
        attn_output = self.o_proj(attn_output)  
        return attn_output, attn_weights  
```
- `attn_output.reshape(...)`: 将计算结果重塑回 `(batch_size, seq_len, hidden_size)` 的形状
- `self.o_proj(attn_output)`: 对输出进行最后的线性变换
- `return ...`: 返回最终的注意力输出和可选的注意力权重

```
class Qwen3DecoderLayer(Qwen2DecoderLayer):
    pass

class Qwen3PreTrainedModel(Qwen2PreTrainedModel):
    pass

class Qwen3Model(Qwen2Model):
    pass

class Qwen3ForCausalLM(Qwen2ForCausalLM):
    def forward(...):
        """... (文档和示例) ..."""
        return super().forward(**super_kwargs)

class Qwen3ForSequenceClassification(Qwen2ForSequenceClassification):
    pass

# ... (其他下游任务模型)
```
- `Qwen3DecoderLayer`: Transformer 的解码器层。直接继承 Qwen2，但它内部会使用我们上面定义的 `Qwen3Attention` 和 `Qwen3MLP`
- `Qwen3Model`: 基础的 Transformer 模型（不带语言模型头）。直接继承 Qwen2
- `Qwen3ForCausalLM`: 用于**因果语言建模**（文本生成）的完整模型。它也继承自 Qwen2 的同名类，只是重写了 `forward` 方法的文档字符串，以提供 Qwen3 的使用示例。实际的计算逻辑通过 `super().forward()` 完全复用了父类的代码
- `Qwen3ForSequenceClassification` 等：用于下游任务（如分类、问答）的模型，也全部直接继承自 Qwen2 的对应实现

---
# .\transformers\models\qwen3\modeling_qwen3.py

```
@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
```
- `@use_kernel_forward_from_hub("RMSNorm")`: 这是一个装饰器，它尝试从 Hub 下载并使用一个预编译的、更高效的 CUDA 核（kernel）来执行 `RMSNorm` 的前向传播计算，以提升性能
- `class Qwen3RMSNorm(nn.Module)`: 定义一个名为 `Qwen3RMSNorm` 的类，它继承自 PyTorch 的基础模块 `nn.Module`

```
def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
```
- `self.weight = nn.Parameter(torch.ones(hidden_size))`: 创建一个可学习的权重参数 `weight`，其形状为 `(hidden_size,)`，并初始化为全 1。`nn.Parameter` 会将其注册为模型参数
- `self.variance_epsilon = eps`: 设置一个很小的数 `eps` (epsilon)，用于在计算中防止除以零

```
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```
- `forward`: 定义该模块的前向传播逻辑
- `input_dtype = hidden_states.dtype`: 保存输入张量的数据类型（如 `float16`）
- `hidden_states = hidden_states.to(torch.float32)`: 为了计算精度，将输入张量转换为 32 位浮点数
- `variance = hidden_states.pow(2).mean(-1, keepdim=True)`: 计算均方值（Root Mean Square 的一部分）。`.pow(2)` 计算平方，`.mean(-1, keepdim=True)` 沿着最后一个维度求平均值
- `hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)`: 进行归一化。`torch.rsqrt` 计算平方根的倒数。这行代码等价于 `hidden_states / sqrt(variance + eps)`
- `return self.weight * hidden_states.to(input_dtype)`: 将归一化后的张量乘以可学习的 `weight`，然后转换回原始的输入数据类型并返回

```
def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```
- `extra_repr`: 一个辅助方法，用于在打印模型结构时提供更详细的表示信息，这里会显示权重的形状和 `eps` 的值

```
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
```
- `self.gate_proj`, `self.up_proj`: 定义两个线性层，它们将输入从 `hidden_size` 映射到 `intermediate_size`。这用于实现 **SwiGLU** 激活函数
- `self.down_proj`: 定义一个线性层，将维度从 `intermediate_size` 映射回 `hidden_size`
- `self.act_fn`: 从 `ACT2FN` 字典中根据配置 `config.hidden_act`（例如 "silu"）获取对应的激活函数

```
def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```
- `self.act_fn(self.gate_proj(x)) * self.up_proj(x)`: 这是 **SwiGLU** 的核心计算。它对输入 `x` 分别通过 `gate_proj` 和 `up_proj`，然后将 `gate_proj` 的结果通过激活函数，再与 `up_proj` 的结果进行逐元素相乘
- `self.down_proj(...)`: 将 SwiGLU 的计算结果通过 `down_proj` 降维，得到最终输出

```
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```
- `rotate_half`: RoPE 的一个辅助函数
- `x1 = ...`, `x2 = ...`: 将输入张量 `x` 的最后一个维度平分为两半
- `return torch.cat((-x2, x1), dim=-1)`: 将后半部分 `x2` 取反，然后与前半部分 `x1` 拼接，实现特征的“旋转”效果

```
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # ... (文档字符串) ...
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
- `apply_rotary_pos_emb`: 将旋转位置编码应用到 Query (`q`) 和 Key (`k`) 上
- `cos = cos.unsqueeze(...)`, `sin = sin.unsqueeze(...)`: 扩展 `cos` 和 `sin` 的维度，以便它们可以和 `q`、`k` 进行广播（broadcast）计算
- `q_embed = (q * cos) + (rotate_half(q) * sin)`: 这是应用 RoPE 的核心数学公式，它等效于在复数域中将 `q` 乘以一个旋转矩阵
- `k_embed = ...`: 对 `k` 执行同样的操作

```
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # ... (文档字符串) ...
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```
- `repeat_kv`: 用于**分组查询注意力 (Grouped-Query Attention, GQA)**
- 它将 Key 和 Value 的头重复 `n_rep` 次，使其数量与 Query 的头数相匹配，从而可以进行注意力计算

```
def eager_attention_forward(...):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # ...
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```
- `eager_attention_forward`: 标准的、非优化的注意力计算前向传播函数
- `key_states = repeat_kv(...)`: 首先处理 GQA，复制 K 和 V 的头
- `attn_weights = torch.matmul(...)`: 计算注意力分数（Query 和 Key 的点积）
- `if attention_mask is not None`: 如果有注意力掩码（用于忽略 padding 或未来 token），则应用它
- `attn_weights = nn.functional.softmax(...)`: 对分数进行 softmax，得到注意力权重
- `attn_weights = nn.functional.dropout(...)`: 应用 dropout
- `attn_output = torch.matmul(...)`: 将注意力权重与 Value 相乘，得到加权和
- `attn_output = attn_output.transpose(...)`: 调整输出张量的形状

```
class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        # ...
        self.q_proj = nn.Linear(...)
        self.k_proj = nn.Linear(...)
        self.v_proj = nn.Linear(...)
        self.o_proj = nn.Linear(...)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
```
- `self.q_proj`, `self.k_proj`, `self.v_proj`: 用于从输入 `hidden_states` 生成 Query, Key, Value 的线性投影层
- `self.o_proj`: 输出投影层，用于将多头注意力的结果合并并映射回 `hidden_size`
- `self.q_norm`, `self.k_norm`: **这是 Qwen3 的一个特点**。它为 Query 和 Key 增加了独立的 RMSNorm 层，在每个头的维度上进行归一化
- `self.sliding_window`: 根据配置决定是否启用**滑动窗口注意力**

```
def forward(...):
        # ...
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ...
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # ... (KV Cache & Attention calculation)
```
- `query_states = self.q_norm(...)`: 计算 Q，并在投影后、应用 RoPE 前，通过 `q_norm` 进行归一化
- `key_states = self.k_norm(...)`: 同样地，计算并归一化 K
- `value_states = ...`: 计算 V（注意 V 没有归一化）
- `apply_rotary_pos_emb(...)`: 将位置编码应用到 Q 和 K
- 后续代码会处理 KV 缓存，并调用像 `eager_attention_forward` 这样的函数来完成最终的注意力计算。

```
class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(...)
        self.post_attention_layernorm = Qwen3RMSNorm(...)
```
- 一个解码器层由**一个自注意力模块** (`self.self_attn`) 和**一个 MLP 模块** (`self.mlp`) 组成
- `self.input_layernorm`: 注意力模块前的归一化层
- `self.post_attention_layernorm`: MLP 模块前的归一化层

```
def forward(...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(...)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```
`forward`: 定义了标准的 Transformer 解码器层的前向传播流程：
1. **Pre-Norm -> Self-Attention -> Residual Connection (残差连接)**
2. **Pre-Norm -> MLP -> Residual Connection**