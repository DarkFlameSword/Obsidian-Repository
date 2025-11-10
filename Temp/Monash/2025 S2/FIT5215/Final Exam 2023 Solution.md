---
date: 2025-11-10
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# 13
![[Pasted image 20251110182140.png]]
a.
Given a noise variable z sampled from a distribution $P_z$, the generator tries to use this noise to create fake data that are indistinguishable from real data (the original images in the dataset S)

b.
generator (G) : in this min-max gamed, G plays like a criminal who making counterfiet money
criminator (D): in this min-max gamed, D plays like a policeman who trying to distinguish the counterfiet money

c. 
Discriminator Optimization: $MaxD log(D(x))+log(1−D(G(z)))$, where x is a real image, and $G(z)$ is a fake image produced by the generator. Generator Optimization: $Min_G log(1−D(G(z))) \;\text{or}\; Max_G log(D(G(z)))$. This objective encourages $G$ to produce images that are classified as real by $D$

d.
**Why?**
At the Nash equilibrium point in a perfectly trained GAN, the discriminator $D^*$ cannot distinguish between real and fake images; hence, it assigns a probability of 0.5 to both.
This outcome implies that the generator $G^*$ has become so good at mimicking the real data distribution that its output is indistinguishable from real data
**Status at Equilibrium Point:**
Optimal Discriminator: It's equally likely to classify real and fake images as real, indicating that it can no longer improve its accuracy (hence, the output is 0.5 for all inputs)
Optimal Generator: It perfectly replicates the real data distribution, meaning the images it generates are as realistic as actual images from the dataset

# 14
![[Pasted image 20251110185026.png]]
a.
32: This number represents the number of filters in the convolutional layer. Each filter is a small matrix
(3,3): This is the size of each filter in the convolutional layer. It indicates that each filter is a 3x3 matrix

![[Pasted image 20251110185227.png]]
b.
The depth of the output is determined by the number of filters. Additionally, we can see that the model has Conv2D, BatchNormalization, MaxPool2D Layers, all do not change the number of depth, since the depth is 32.

![[Pasted image 20251110185934.png]]
c.
![[78A839B53707F644028D465A2D25CFDA.jpg]]
![[Pasted image 20251110190918.png]]
Batch Normalization:
- Reduce the Internal covariant shift, to improve the robutness of the model and avoid overfitting
- Converge faster by using bigger learning rate
- Reduce overfitting
- Make training more stable

Max Pooling:
- Reduce the number of training parameters, releasing computational cost
- Extract dominant features

# 15
![[Pasted image 20251110191529.png]]
![[62150644310ae1e2dfef76f766d75797_720.png]]

# 19
![[Pasted image 20251110192405.png]]
![[Pasted image 20251110192652.png]]
a.
Process X represents "Word Embedding", mapping the real words to tensor without losing semantics

![[Pasted image 20251110192754.png]]
b.
A employer is looking for a candidate with **"Python"** skills
**Without Embeddings:** The hiring software scans resumes only for the exact word "Python." It might reject a highly qualified candidate who listed their experience as **"Django development"** (a popular Python web framework) but didn't explicitly write the word "Python" often enough

![[Pasted image 20251110193415.png]]
c.
word2Vec
doc2Vec

![[Pasted image 20251110193620.png]]
d.
An application that utilizes the many-to-one RNN model is Sentiment Analasis. In this application, the RNN takes a sequence of words as input and predicts the probability of the sentiment (positive or negative)

# 20
![[Pasted image 20251110193834.png]]
a.
Name: Average loss function
Purpose: The purpose of this term is to minimize the error of the neural network's predictions on the training data

b.
Name: The regularization term
Purpose: To prevent overfitting by regularize extrem weights in the model parameters $\theta$

# 21
![[Pasted image 20251110195226.png]]
- **Receptive Field:** As you move deeper into the network (through successive convolution and pooling layers), each neuron "sees" a larger portion of the original input image
- **Hierarchical Composition:** Lower layers detect simple, local patterns (like edges) because their receptive fields are small. Higher layers combine these simple patterns from previous layers across a wider spatial area, allowing them to recognize complex, global structures (like entire objects)

# 23
![[Pasted image 20251110195456.png]]
a.
The training outcome shows signs of overfitting. Overfitting occurs when the model performs well on the training data but does not generalize well to new, unknown data. This can be inferred from the fact that while the training accuracy is increasing and the training loss is decreasing consistently, the validation accuracy is not improving much, and the validation loss starts to fluctuate or even increase after a certain point. 
Specifically, from the data shown, around epoch 10, the model's validation loss is not showing improvements, while the training loss is still decreasing

![[Pasted image 20251110195650.png]]
b.
Dropout:
Stop training early: Stop training when performance on the validation set begins to decline to prevent the model from continuing to learn from noise in the training data

Early Stopping:
During training, a portion of neurons are randomly dropped to force the model to learn more robust features and prevent the model from becoming overly reliant on certain specific neurons
