---
date: 2025-10-12
author:
  - Siyuan Liu
tags:
  - papper
---
flagship model: `Qwen3-235B-A22B`
- 235B: 235 **B**illion total parameters
- A22B: **A**ctivate 22 **B**illion parameters when answering a question

# 特点一：thinking mode与non-thinking mode结合无需切换
This flexibility ensures that developers and users can adapt the model's behavior to suit specific tasks efficiently
## continual supervised fine-tuning (SFT)
To achieve this, we conduct continual supervised fine-tuning (SFT) on the Reasoning RL model and design a chat template to fuse the two modes. Moreover, we find that models capable of handling both modes proficiently perform consistently well under different thinking budgets.

# 特点二：添加thinking budgets mechanism
providing users with fine-grained control over the level of reasoning effort applied by the model during task execution

# 特点三：Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts
128 total experts with 8 activated experts per token

# 特点四：119 languages and dialects, with a total of 36 trillion tokens

----
# Qwen3 base model训练过程
## three-stage pre-training strategy
### the first stage
the model is trained on about 30 trillion tokens to build a strong foundation of general knowledge

### the second stage
it is further trained on knowledge-intensive data to enhance reasoning abilities in areas like science, technology, engineering, and mathematics (STEM) and coding

### the third stage
the model is trained on long-context data to increase its maximum context length from 4,096 to 32,768 tokens

---
## post-training
![[Pasted image 20251012165426.png]]
- (1) Thinking Control: This involves the integration of two distinct modes, namely the "non-thinking" and "thinking" modes, providing users with the flexibility to choose whether the model should engage in reasoning or not, and to control the depth of thinking by specifying a token budget for the thinking process
- (2) Strong-to-Weak Distillation: This aims to streamline and optimize the post-training process for lightweight models. By leveraging the knowledge from large-scale models, we substantially reduce both the computational costs and the development efforts required for building smallerscale models
### the first and second stage
focus on developing strong reasoning abilities through long chain-of-thought (CoT) cold-start finetuning and reinforcement learning focusing on mathematics and coding tasks
#### Long-CoT Cold Start
#### Reasoning RL

#### Thinking Mode Fusion

### the third and third stage
we combine data with and without reasoning paths into a unified dataset for further fine-tuning, enabling the model to handle both types of input effectively, and we then apply generaldomain reinforcement learning to improve performance across a wide range of downstream tasks
#### General RL
To provide feedback for the aforementioned tasks, we utilized three distinct types of rewards:
- (1) Rule-based Reward: The rule-based reward has been widely used in the reasoning RL stage, and is also useful for general tasks such as instruction following (Lambert et al., 2024) and format adherence. Well-designed rule-based rewards can assess the correctness of model outputs with high precision, preventing issues like reward hacking. 
- (2) Model-based Reward with Reference Answer: In this approach, we provide a reference answer for each query and prompt Qwen2.5-72B-Instruct to score the model's response based on this reference. This method allows for more flexible handling of diverse tasks without requiring strict formatting, avoiding false negatives that can occur with purely rule-based rewards. 
- (3) Model-based Reward without Reference Answer: Leveraging human preference data, we train a reward model to assign scalar scores to model responses. This approach, which does not depend on a reference answer, can handle a broader range of queries while effectively enhancing the model's engagement and helpfulness.

#### Strong-to-Weak Distillation
- (1) Off-policy Distillation: At this initial phase, we combine the outputs of teacher models generated with both /think and /no think modes for response distillation. This helps lightweight student models develop basic reasoning skills and the ability to switch between different modes of thinking, laying a solid foundation for the next on-policy training phase
- (2) On-policy Distillation: In this phase, the student model generates on-policy sequences for fine-tuning. Specifically, prompts are sampled, and the student model produces responses in either /think or /no think mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence



---
# 主要技术引用
- Grouped Query Attention
- SwiGLU
- Rotary Positional Embeddings
- RMSNorm
- remove QKV-bias used in Qwen2， introduce QK-Norm
- global-batch load balancing loss：encourage expert specialization
- ABF
- YARN
- Dual Chunk Attention