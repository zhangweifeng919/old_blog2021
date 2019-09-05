---
title: 使用Pytorch实现word2vec(skip-gram)
date: 2019-08-03 11:08:23
tags:
- NLP
- word2vec
- Pytorch
categories:
- 文档阅读笔记
description: 学习使用Pytorch实现word2vec,翻译于：https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
---
## 引入相关包
```PYTHON
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

```
## 语料
为了跟踪每一个步骤，使用了下面非常小的语料
```PYTHON
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]
```
## 创建词汇表
创建词汇表是word2vec的第一步，因为这个词汇表不支持扩展，所以在一开始就要创建。
这个语料库非常简短，在实际操作中可能需要标准化，去掉一些标点符号什么的，现在我们用的语料比较简单，现在我们把它令牌化。
```PYTHON
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(corpus)
```
他会输出这样一个列表
```PYTHON
[['he', 'is', 'a', 'king'],
 ['she', 'is', 'a', 'queen'],
 ['he', 'is', 'a', 'man'],
 ['she', 'is', 'a', 'woman'],
 ['warsaw', 'is', 'poland', 'capital'],
 ['berlin', 'is', 'germany', 'capital'],
 ['paris', 'is', 'france', 'capital']]
```
现在开始迭代它，生成一个不重复的单词列表，然后建立两个字典，来映射单词和索引。
```PYTHON
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)
print(idx2word)
```
这个将输出
```PYTHON
0: 'he',
 1: 'is',
 2: 'a',
 3: 'king',
 4: 'she',
 5: 'queen',
 6: 'man',
 7: 'woman',
 8: 'warsaw',
 9: 'poland',
 10: 'capital',
 11: 'berlin',
 12: 'germany',
 13: 'paris',
 14: 'france'
```
现在我们可以生成中心词和上下文词的组合，假设上下文窗口是对称的并且大小为2.
```PYTHON
window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array

```
它会得到一个中心词和上下词的组合：
```PYTHON
array([[ 0,  1],
       [ 0,  2],
       ...
```
可以很简单转换成单词：
```PYTHON
he is
he a
is he
is a
is king
a he
a is
a king
```
具有完美的意义：
<img src="/images/文章阅读笔记/使用pytorch实现word2vec_skip_gram/图一.png" style="width:30%" />
## 定义目标
现在我们详细介绍从第一个方程到最后实现的细节

对于skip-gram 我们感兴趣的是给定中心词和一些参数来预测上下文词。这是我们的对于一个组合的概率分布
$$
P(context | center,\theta)
$$
现在我们想最大化所有的中心词和上下文词的组合：
$$
\max \prod_{center} \prod_{context} P(context | center,\theta)
$$
等等，为什么？
因为我们感兴趣的是给定中心词来预测上下文词，对于每个上下文词和中心词的组合，我们想最大化$P(context | center,\theta)$,所有的概率相加等于1——对于所有不存在的中心词，上下文词组合，我们默认他们为0。通过把这些概率相乘，如果我们的模型比较好，结果就会接近1，如果比较差，结果就会接近0，我们追求好的模型，所以，这是一个最大化的操作。
这个表达式不太适合计算，这就是为什么进行一些非常常规的变换。
### 使用负对数似然替换概率
回忆一下，在神经网络中，要最小化损失函数。我们可以简单的将P乘以-1，应用对数可以给我们更好的计算性质。对数不改变函数的极值，因为对数是严格的单调函数。所以表达式可以变为：
$$
\min_{\theta} - \log \prod_{center} \prod_{context} P(context | center,\theta)
$$
### 用加法替换乘法
下一步就是用加号替换乘号，因为：
$$
\log (a * b)= \log a + \log b
$$
### 变换成合适的损失函数
$$
loss = - \cfrac{1}{T} \sum_{center} \sum_{context} \log P(context | center,\theta)
$$
### 定义P
很好，但是我们怎么定义$P(context | center)$目前为止，假设我们接触到的单词有两个向量，第一个是中心词$(V)$,第二个是上下文文词$(U)$,P的定义看起来像下面那样：
$$
P(\text { context } | \text { center })=\frac{\exp \left(u_{\text { context }}^{T} v_{\text { center }}\right)}{\sum_{w \in \text { vocab }} \exp \left(u_{w}^{T} v_{\text { center }}\right)}
$$
真可怕！
让我们把它分解成小块，看下面的结构：
$$
\frac{\exp (\cdot)}{\sum \exp (\cdot)}
$$
这是softmax函数，再仔细看一下分子：
$$
u_{\text { context }}^{T} v_{\text { center }}
$$
U和V都是向量，这个表达式是中心词和上下文词的标量积。他们越相似，就越大。
现在看分母：
$$
\sum_{w \in \text { vocab }}
$$
我们遍历了词汇表的所有单词
$$
u_{w}^{T} v_{\text { center }}
$$
然后计算中心词与词表中的每个词作为上下文词的相似性。
### 总结一下
对于每个中心词和上下文词组合，计算他们的”相似分数“，然后除以每个理论上可能的上下文——知道分数相对的高还是低。softmax 可以保证值的范围在0和1之间。定义了合格的概率分布。
## 很好，我们现在来编码
神经网络实现这个概念，用了三层网络，输入层，隐含层，输出层。
### 输入层
输入层是one-hot编码的中心词，它的维度是[1,vocabulary_size]
```PYTHON
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x
```
### 隐含层
隐含层作为我们的V向量，所以它有embedding_dim个神经元，为了计算它，我们需要W1权重矩阵。当然，它需要有[embedding_size,vocabulary_size]的形状。这里没有激活函数，只需要简单的矩阵相乘。
```PYTHON
embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
z1 = torch.matmul(W1, x)
```
重要的是——W1的每列保存着每个单词的v向量。为什么？因为x是one-hot向量，如果矩阵乘以one-hot变量，相当于从矩阵中选择一列，你可以使用一张纸验算一下。
### 输出层
最后一层要有vacalbuary_size个神经元，因为它为每个单词生成概率，因此W2是[vocalbuary_size,embedding_size]形状。
```PYTHON
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
z2 = torch.matmul(W2, z1)
```
在上面我们使用softmax层，Pytorch 提供了相应的版本。和log 结合在一起，因为常规的softmax 不是数值稳定的。
```PYTHON
log_softmax = F.log_softmax(a2, dim=0)
```
这个算式，是在做softmax之后，再应用对数。
现在我们可以计算loss,Pytorch 也提供了函数
```PYTHON
loss = F.nll_loss(log_softmax.view(1,-1), y_true)
```
这个nll_loss在log softmax上计算nagtive-log-likelihood ,y_true是上下文词，我们想让这个越高越好，因为x,y_true组合是真实的中心词和上下文词。
### 反向传播
我们完成了正向传递，下面进行反向传播
```PYTHON
loss.backward()
```
使用SDG进行优化，这个很简单，直接手写比创建优化器对象更加简单
```PYTHON
W1.data -= 0.01 * W1.grad.data
W2.data -= 0.01 * W2.grad.data
```
最后一步就是把梯度置为0，确保下次循环不受影响
```PYTHON
W1.grad.data.zero_()
W2.grad.data.zero_()
```
### 训练循环
```PYTHON
embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 100
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

```
一个可能棘手的问题是y_true定义。我们没有明确创建one-hot,nll_loss 会自动帮我们完成。
```PYTHON
Loss at epo 0: 4.241989389487675
Loss at epo 10: 3.8398486052240646
Loss at epo 20: 3.5548086541039603
Loss at epo 30: 3.343840673991612
Loss at epo 40: 3.183084646293095
Loss at epo 50: 3.05673006943294
Loss at epo 60: 2.953996729850769
Loss at epo 70: 2.867735825266157
Loss at epo 80: 2.79331214427948
Loss at epo 90: 2.727727291413716
Loss at epo 100: 2.6690095041479385
```
## 提取向量
现在我们训练了一个网络，最后一件事就是提取每个单词的向量，这里有三个可能的方式
    - 使用W1的v向量
    - 使用W2的u向量
    - 使用u和v的平均
你可以自己思考什么时候用哪个

## 全部代码
```PYTHON

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]

def tokenize_corpus(corpus):
    tokens=[x.split() for x in corpus]
    return tokens


tokenized_corpus=tokenize_corpus(corpus)
print(tokenized_corpus)

vocabulary=[]
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)
word2idx={w:idx for (idx,w) in enumerate(vocabulary)}
idx2word={idx:w for (idx,w) in enumerate(vocabulary)}

vocabulary_size=len(vocabulary)

print(idx2word)
window_size=2
idx_pairs=[]
for sentence in tokenized_corpus:
    indices=[word2idx[word] for word in sentence]
    for center_word_pos in range(len(indices)):
        for w in range(-window_size,window_size+1):
            context_word_pos=center_word_pos+w
            if context_word_pos<0 or context_word_pos>= len(indices) or center_word_pos==context_word_pos:
                continue
            context_word_idx=indices[context_word_pos]
            idx_pairs.append([indices[center_word_pos],context_word_idx])
idx_pairs=np.array(idx_pairs)
print(idx_pairs)

def get_input_layer(word_idx):
    x=torch.zeros(vocabulary_size).float()
    x[word_idx]=1.0
    return x

embedding_size=5
W1=Variable(torch.randn(embedding_size,vocabulary_size).float(),requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_size).float(), requires_grad=True)
num_epochs = 100
learning_rate = 0.001
for epo in range(num_epochs):
    loss_val=0
    for data,target in idx_pairs:
        x= Variable(get_input_layer(data)).float()
        y_true=Variable(torch.from_numpy(np.array([target])).long())
        z1=torch.matmul(W1,x)
        z2=torch.matmul(W2,z1)
        log_softmax=F.log_softmax(z2,dim=0)
        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

```




















