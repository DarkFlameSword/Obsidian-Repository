---
date: 2025-09-08
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Gredient Back Propagation
![[Pasted image 20250908124658.png]]
## From Loss to $h_3$
$$l = (x;\theta) = -log \frac{e^{h_3}}{e^{h_1}+e^{h_2}+e^{h_3}} =  -\left[log (e^{h_3}) - log(e^{h_1}+e^{h_2}+e^{h_3}) \right] = -h_3 + log(e^{h_1}+e^{h_2}+e^{h_3})$$ $$\begin{aligned}
& \frac{\partial l}{\partial{h_3}}  = -1 + log(e^{h_1}+e^{h_2}+e^{h_3})' \\
& = -1+ \frac{d}{du}log(u) \times \frac{d}{dh_3}(e^{h_1}+e^{h_2}+e^{h_3}) \\
& = -1 + \frac{u'}{u} \times 1 \\
& = -1 + \frac{1}{e^{h_1}+e^{h_2}+e^{h_3}}\\
\end{aligned}$$
# From loss to W3 , b 3
$$h_3 = h_2W_3 + b_3$$

$$\frac{\partial l}{\partial{W_3}} = \frac{\partial l}{\partial{h_3}} \cdot \frac{\partial h_3}{\partial{W_3}}$$
$$\begin{aligned}
& \text{since we already got}\frac{\partial l}{\partial{W_3}}\text{before, so here is how to get the rest part:}\\
& \frac{\partial h_3}{\partial{W_3}} = h_2\\
& \frac{\partial l}{\partial{W_3}} = h_2^T \cdot \left(-1 + \frac{1}{e^{h_1}+e^{h_2}+e^{h_3}} \right)\\
& \text{as the same process that we can get} \;b_3:\\
& \frac{\partial l}{\partial{b_3}} =  \left(-1 + \frac{1}{e^{h_1}+e^{h_2}+e^{h_3}} \right)
\end{aligned}$$

# Why does deep learning need GPU and TPU?
consider below formula:
$$
\frac{\partial l}{\partial W^1} = \frac{\partial l}{\partial h^3} \cdot \frac{\partial h^3}{\partial h^2} \cdot \frac{\partial h^2}{\partial \bar{h^2}} \cdot \frac{\partial \bar{h^2}}{\partial h^1} \cdot \frac{\partial h^1}{\partial \bar{h^1}} \cdot \frac{\partial h^1}{\partial W^1}
= (h^0)^T (p-1_y)(W^3)^T diag(\sigma'(h^2))(W^2)^T diag(\sigma'(h^1))
$$
For a deep net, this back probagation requires many matrix multiplications
- We need specific hardware that can parallel and significantly speed up matrix multiplication operation
- GPU(Graphic processing Unit) and TPU(Tensor Processing Unit)