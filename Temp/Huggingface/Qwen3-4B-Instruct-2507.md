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
