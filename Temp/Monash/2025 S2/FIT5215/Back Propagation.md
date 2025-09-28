---
date: 2025-09-08
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Gredient Back Propagation
## From Loss to $h_3$
![[Pasted image 20250908124658.png]]
$$l = (x;\theta) = -log \frac{e^{h_3}}{e^{h_1}+e^{h_2}+e^{h_3}} =  -\left[log (e^{h_3}) - log(e^{h_1}+e^{h_2}+e^{h_3}) \right] = -h_3 + log(e^{h_1}+e^{h_2}+e^{h_3})$$ $$\frac{\partial l}{\partial{h_3}} = -1 + log(e^{h_1}+e^{h_2}+e^{h_3})' = -1+ \frac{d}{du}log(u) \times \frac{d}{dh_3}(e^{h_1}+e^{h_2}+e^{h_3}) = -1 + \frac{u'}{u} \times 1 = -1 + \frac{1}{e^{h_1}+e^{h_2}+e^{h_3}}$$
# From loss to W3 , b 3
$$h_3 = h_2W_3 + b_3$$
$$$$

# Why does deep learning need GPU and TPU?
![[Pasted image 20250908130933.png]]