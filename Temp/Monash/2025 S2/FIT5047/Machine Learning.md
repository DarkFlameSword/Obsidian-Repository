---
date: 2025-10-09
author:
  - Siyuan Liu
tags:
  - FIT5047
---
# Inductive Learning
**Experimentation cycle:**
1. Learn model parameters on Training Data
2. Fine tune the model on Validation (Held-out) Data
3. Compute performance on Test Data

# Evaluation Metrix
混淆矩阵中的四个量
- TP（True Positive）：实际为正，预测为正
- FP（False Positive）：实际为负，预测为正
- FN（False Negative）：实际为正，预测为负
- TN（True Negative）：实际为负，预测为负
- 
公式：
- Accuracy（准确率）= $\frac{TP + TN}{TP + FP + FN + TN}$
- Recall（召回率、灵敏度）= $\frac{TP}{TP + FN}$
- Precision（精确率）= $\frac{TP}{TP + FP}$
- F1-score（F1 值）= $\frac{2 × (Precision × Recall) }{Precision + Recall}$

