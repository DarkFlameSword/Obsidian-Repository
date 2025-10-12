---
date: 2025-10-10
author:
  - Siyuan Liu
tags:
  - Huggingface
---
```
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
```
- AutoProcessor：自动加载与模型匹配的处理器（既包含分词器 tokenizer，也包含图像预处理器 image processor，以及聊天模板 chat template）

```
# default: Load the model on the available device(s)
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-30B-A3B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
```
- 从 Hugging Face Hub 拉取名为 Qwen/Qwen3-VL-30B-A3B-Instruct 的权重与配置。
- dtype="auto"：自动选择合适的精度（如 bfloat16/float16），以兼顾速度与显存。
- device_map="auto"：自动将模型放到可用设备（单/多块 GPU、CPU）上并做分片加载。需要 accelerate 支持。
- 部分模型可能需要 trust_remote_code=True
- attn_implementation="flash_attention_2"：使用 FlashAttention v2，提升速度和节省显存（尤其多图/视频场景）。

```
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
```
加载与该模型配套的处理器，内含：
- 文本分词规则、特殊标记
- 图像预处理（尺寸、归一化等）
- 多模态聊天模板（把“角色+内容”的结构转成模型可读的序列）

```
messages = [
    {
        "role": "user", # 表示用户消息
        "content": [ # 包含两个片段
            {
                "type": "image", # 一个图片片段
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg", # image 给出图片的 URL。处理器会下载并预处理该图像
            },
            {
	            "type": "text", # 一个文本片段
	            "text": "Describe this image." # 具体指令是"Describe this image."
	        },
        ],
    }
]
```
- 定义一轮对话
- 该结构符合多模态聊天模板的预期格式。若在离线/内网环境，可改为本地路径或 PIL.Image

```
# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
```
- apply_chat_template：把 messages 按模型的聊天约定拼接为提示词（prompt），并处理图像
- tokenize=True：将文本转为 token ids（input_ids 等）；同时会生成图像张量等多模态输入
- add_generation_prompt=True：在提示词末尾追加“助理开始回答”的标记/前缀，引导模型生成
- return_dict=True：返回 Python 字典，便于用 inputs 直接传给 model.generate
- return_tensors="pt"：返回 PyTorch 张量（如 input_ids、attention_mask、以及图像相关张量）。
- 最终 inputs 典型包含：input_ids、attention_mask、（多模态键如 images/pixel_values、image_sizes 等，具体视处理器实现而定）

```
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
```
- model.generate: 调用生成接口，基于输入产生最多 128 个新 token（续写长度上限）
- 未显式指定采样策略时，通常为贪心或模型 config 的默认策略。可按需添加 temperature、top_p、num_beams 等参数

```
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
```
- model.generate 返回的是“输入 + 生成”的拼接序列。该列表推导式对每个样本把前面等长的输入部分切掉，只保留“新生成的” token。
- zip(inputs.input_ids, generated_ids) 可支持批处理（这里只有 1 条消息，但写法对 batch 通用）

```
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
- 将 token id 序列解码为字符串
- skip_special_tokens=True：跳过如 <|im_start|>、<|endoftext|> 等特殊标记
- clean_up_tokenization_spaces=False：保留空格细节，避免自动清理带来的格式变化
- 返回列表（批大小个数）。单条输入时可用 output_text\[0\] 取字符串