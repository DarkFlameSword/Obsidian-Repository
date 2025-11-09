---
date: 2025-11-09
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Encoder-decoder model for seq2seq
![[Pasted image 20251109122548.png]]
# Two strategies for trainning
## Greedy decoding
![[Pasted image 20251109122658.png]]

---
## Beam search
![[Pasted image 20251109122706.png]]
![[Pasted image 20251109122805.png]]
# Drawback of fixed context
- Fixed context vector 𝒄 is easily overwhelmed by long inputs or long outputs
- At a specific timestep 𝑗, some words or items in the input sequence might possibly contribute more to the generation of next item or word in the output sequence