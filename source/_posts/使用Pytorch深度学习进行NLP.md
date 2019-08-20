---
title: 使用Pytorch深度学习进行NLP
date: 2019-07-22 11:20:57
tags:
- nlp
- Pytorch
categories:
- 文档阅读笔记
description: 对pytorch 官网上Deep learning for NLP with pytorch章节的翻译，链接https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html
---
作者：Robert Guthrie
## 前言
这个教程会教您使用pytorch进行深度学习编程的关键思想。这里有很多概念，比如图计算和自动梯度计算并不是pytorch独有的，在其他的深度学习的工具中也同样存在。
我写这个教程主要针对NLP问题，和教那些之前没有用过相关的深度学习框架（比如 TensorFlow, Theano, Keras, Dynet）的同学。假设我们需要解决的问题是NLP中的核心问题，比如词性标注，语言模型等，也假设我们已经在介绍人工智能的课程中熟悉了神经网络，通常这些课程包含了反向传播算法和前馈神经网络。注意到它们是线性和非线性的链组成的。假设你已经有了必要的知识储备，这个教程将会教你开始写深度学习的代码。
注意：教程主要关注点在于模型，不是数据，所以我在例子中会使用少量的低纬度数据，你可以看到当训练时，权重是如何改变的。如果你想测试你的真实数据，你可以复制模型然后使用它。
## pytorch 介绍
### 介绍torch 的tensor库
深度学习是在tensor上计算的。tensor是矩阵的一般化，可以被2个以上索引。我们之后会看到这个到底什么意思，现在我们可以使用tensor 做些什么。
```PYTHON
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) # 设置随机数生成器的种子
```
### 创建Tensors
使用torch.Tensor()函数可以从Python数组中创建Tensors
```PYTHON
# torch.tensor(data) creates a torch.Tensor object with the given data.
V_data = [1., 2., 3.]
V = torch.tensor(V_data)
print(V)

# Creates a matrix
M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.tensor(M_data)
print(M)

# Create a 3D tensor of size 2x2x2.
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.tensor(T_data)
print(T)
```
Out:
```PYTHON
tensor([1., 2., 3.])
tensor([[1., 2., 3.],
        [4., 5., 6.]])
tensor([[[1., 2.],
         [3., 4.]],

        [[5., 6.],
         [7., 8.]]])
```
什么是3D tensor ? 设想一下，如果你有一个向量，索引这个向量你会得到一个标量，如果你有一个矩阵，索引这个矩阵你会得到一个向量，如果你有一个3D tensor，索引这个tensor 你会得到一个矩阵！
关于术语的注意：当我说tensor时，指的是torch.tensor对象，向量和矩阵是torch中的一维和二维特殊例子。当我提及3Dtensor时，我指的就是3D tensor。
```PYTHON
# Index into V and get a scalar (0 dimensional tensor)
print(V[0])
# Get a Python number from it
print(V[0].item())

# Index into M and get a vector
print(M[0])

# Index into T and get a matrix
print(T[0])
```
Out
```PYTHON
tensor(1.)
1.0
tensor([1., 2., 3.])
tensor([[1., 2.],
        [3., 4.]])
```
你也可以创建其他类型的tensor,默认情况下的类型是float，为了创建整数类型，你可以使用torch.LongTensor()函数，你可以参考其他文档了解更多的数据类型，但是一般情况下Float和Long 是最常使用的类型。
你可以使用torch.randn()创建指定维度的随机的数据。
```PYTHON
x = torch.randn((3, 4, 5))
print(x)
```
Out:
```PYTHON
tensor([[[-1.5256, -0.7502, -0.6540, -1.6095, -0.1002],
         [-0.6092, -0.9798, -1.6091, -0.7121,  0.3037],
         [-0.7773, -0.2515, -0.2223,  1.6871,  0.2284],
         [ 0.4676, -0.6970, -1.1608,  0.6995,  0.1991]],

        [[ 0.8657,  0.2444, -0.6629,  0.8073,  1.1017],
         [-0.1759, -2.2456, -1.4465,  0.0612, -0.6177],
         [-0.7981, -0.1316,  1.8793, -0.0721,  0.1578],
         [-0.7735,  0.1991,  0.0457,  0.1530, -0.4757]],

        [[-0.1110,  0.2927, -0.1578, -0.0288,  0.4533],
         [ 1.1422,  0.2486, -1.7754, -0.0255, -1.0233],
         [-0.5962, -1.0055,  0.4285,  1.4761, -1.7869],
         [ 1.6103, -0.7040, -0.1853, -0.9962, -0.8313]]])
```
### 操作tensor
你可以使用你喜欢的方式操作tensor
```PYTHON
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
z = x + y
print(z)
```
Out:
```PYTHON
tensor([5., 7., 9.])
```
你可以查看这个[文档]()获取完整的操作列表，它不仅仅只包含了数学的运算。
其中一个在后面会用到的操作就是连接。
```PYTHON
# By default, it concatenates along the first axis (concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# second arg specifies which axis to concat along
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# If your tensors are not compatible, torch will complain.  Uncomment to see the error
# torch.cat([x_1, x_2])
```
Out
```PYTHON
tensor([[-0.8029,  0.2366,  0.2857,  0.6898, -0.6331],
        [ 0.8795, -0.6842,  0.4533,  0.2912, -0.8317],
        [-0.5525,  0.6355, -0.3968, -0.6571, -1.6428],
        [ 0.9803, -0.0421, -0.8206,  0.3133, -1.1352],
        [ 0.3773, -0.2824, -2.5667, -1.4303,  0.5009]])
tensor([[ 0.5438, -0.4057,  1.1341, -0.1473,  0.6272,  1.0935,  0.0939,  1.2381],
        [-1.1115,  0.3501, -0.7703, -1.3459,  0.5119, -0.6933, -0.1668, -0.9999]])
```
### 重排tensor形状
使用.view()函数可以改变tensor的形状，这个函数经常使用，因为神经网络的输入格式是确定的。经常需要改变输入的形状
后才满足输入的格式要求
```PYTHON
x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))  # Reshape to 2 rows, 12 columns
# Same as above.  If one of the dimensions is -1, its size can be inferred
print(x.view(2, -1))
```
Out:
```PYTHON
tensor([[[ 0.4175, -0.2127, -0.8400, -0.4200],
         [-0.6240, -0.9773,  0.8748,  0.9873],
         [-0.0594, -2.4919,  0.2423,  0.2883]],

        [[-0.1095,  0.3126,  1.5038,  0.5038],
         [ 0.6223, -0.4481, -0.2856,  0.3880],
         [-1.1435, -0.6512, -0.1032,  0.6937]]])
tensor([[ 0.4175, -0.2127, -0.8400, -0.4200, -0.6240, -0.9773,  0.8748,  0.9873,
         -0.0594, -2.4919,  0.2423,  0.2883],
        [-0.1095,  0.3126,  1.5038,  0.5038,  0.6223, -0.4481, -0.2856,  0.3880,
         -1.1435, -0.6512, -0.1032,  0.6937]])
tensor([[ 0.4175, -0.2127, -0.8400, -0.4200, -0.6240, -0.9773,  0.8748,  0.9873,
         -0.0594, -2.4919,  0.2423,  0.2883],
        [-0.1095,  0.3126,  1.5038,  0.5038,  0.6223, -0.4481, -0.2856,  0.3880,
         -1.1435, -0.6512, -0.1032,  0.6937]])
```
### 计算图和自动微分
计算图的概念对于深度学习的编程是非常重要的，因为它不需要你自己编写反向传播梯度。一个计算图是如何把你的数据和输出结合起来的简易说明书。因为计算图包含了参数和相应的操作，使得有足够的信息计算导数。这可能听起来有点模糊。所以让我们看看使用了基本的标志requires_grad后会发生什么。
首先，从程序员的角度思考，在torch.Tensor对象中，我们保存了什么？很明显，保存了数据，形状和一些其他的东西。我们把两个tensor相加得到一个输出的tensor。这个输出的tensor知道自己的数据和形状，但是它不知道自己是两个tensor相加得到的。（它可能是从文件里读取的，也可能是其他操作得到的，等等）
如果把requires_grad=true，tensor将会记录下来，它自己是如何得到的。
```PYTHON
# Tensor factory methods have a ``requires_grad`` flag
x = torch.tensor([1., 2., 3], requires_grad=True)

# With requires_grad=True, you can still do all the operations you previously
# could
y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
print(z)

# BUT z knows something extra.
print(z.grad_fn)
```
Out:
```PYTHON
tensor([5., 7., 9.], grad_fn=<AddBackward0>)
<AddBackward0 object at 0x7f1b28628d68>
```
所以Tensor就知道自己是如何得来的。不是从文件中来的，也不是乘法或者除法得来的，如果你继续跟踪z.grad_fn,你会发现你是由x和y得来的。
但是它是如何帮助我们计算梯度的呢？
```PYTHON
# Lets sum up all the entries in z
s = z.sum()
print(s)
print(s.grad_fn)
```
Out:
```PYTHON
tensor(21., grad_fn=<SumBackward0>)
<SumBackward0 object at 0x7f1b28628ac8>
```
那么现在，这个总和对x的导数是什么呢？在数学上，我们想获得：
$$
\cfrac{\partial s}{\partial x}
$$
我们知道s是tensor z 的和，z是x+y得到的。那么
$$
s = \overbrace{x_0 + y_0}^\text{$z_0$} + \overbrace{x_1 + y_1}^\text{$z_1$} + \overbrace{x_2 + y_2}^\text{$z_2$}
$$
所以有了足够的信息得到我们想要的导数就是1.
当然，它掩盖了怎样计算导数，关键点在于它带着足够的信息，让我们可以计算导数，在现实中，开发者编码了如何计算sum和加法的梯度和反向传播算法。再深入的讨论就超出了本教程的范围。
让我们使用pytorch计算梯度和验证我们是正确的。（注意，如果你多次运行这段代码，梯度会增加，这是因为pytorch累加了梯度到.grad属性中，这样对于很多模型很方便）
```PYTHON
# calling .backward() on any variable will run backprop, starting from it.
s.backward()
print(x.grad)
```
Out:
```PYTHON
tensor([1., 1., 1.])
```
理解下面的代码段对于成为一个合格的深度学习程序员是重要的。
```PYTHON
x = torch.randn(2, 2)
y = torch.randn(2, 2)
# By default, user created Tensors have ``requires_grad=False``
print(x.requires_grad, y.requires_grad)
z = x + y
# So you can't backprop through z
print(z.grad_fn)

# ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
# flag in-place. The input flag defaults to ``True`` if not given.
x = x.requires_grad_()
y = y.requires_grad_()
# z contains enough information to compute gradients, as we saw above
z = x + y
print(z.grad_fn)
# If any input to an operation has ``requires_grad=True``, so will the output
print(z.requires_grad)

# Now z has the computation history that relates itself to x and y
# Can we just take its values, and **detach** it from its history?
new_z = z.detach()

# ... does new_z have information to backprop to x and y?
# NO!
print(new_z.grad_fn)
# And how could it? ``z.detach()`` returns a tensor that shares the same storage
# as ``z``, but with the computation history forgotten. It doesn't know anything
# about how it was computed.
# In essence, we have broken the Tensor away from its past history
```
Out:
```PYTHON
False False
None
<AddBackward0 object at 0x7f1b28692f60>
True
None
```
你可以通过with torch.no_grad() 包裹代码段，使得.requires_grad=true的代码停止跟踪历史
```PYTHON
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
```
Out:
```PYTHON
True
True
False
```
## 使用pytorch进行深度学习
### 深度学习的基本组成：仿射映射，非线性和目标
深度学习是线性和非线性组合起来的聪明方法。非线性的引入造就了很多强大的模型，在这个章节，我们将会操作这些核心的组件，构造目标函数，观察模型是如何训练的。
### 仿射映射
深度学习的主力之一就是仿射映射，如下：
$$
f(x)=Ax+b
$$
A是矩阵，x和b是向量。A和b是需要学习的参数，b经常叫做偏置项。
Pytorch和其他大部分深度学习框架和传统的代数学有一点不同。它映射输入的是行而不是列。下面输出的第i行是输入的第i行在A下的映射加上偏置项。看下面的例子：
```PYTHON
# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```
```PYTHON
in = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = torch.randn(2, 5)
print(lin(data))  # yes
```
Out:
```PYTHON
tensor([[ 0.1755, -0.3268, -0.5069],
        [-0.6602,  0.2260,  0.1089]], grad_fn=<AddmmBackward>)
```
### 非线性
首先注意下面的事实将会解释为什么我们需要非线性放在第一位。假设我们有两个仿射映射
$f(x)=Ax+b$和$g(x)=Cx+d$,那么$f(g(x))$是什么？
$$
f(g(x)) = A(Cx + d) + b = ACx + (Ad + b)
$$
AC是矩阵，$Ad+b$是向量，所以我们可以看到，把映射组合起来得到新的映射。
从这里我们可以知道，即使我们让神经网络成为仿射成分的长链，然而并不会给模型比仿射映射更强的功能。
如果我们在两个仿射映射层之间中引入非线性，情况就不会像上面那样，我们可以建造更强大的模型。
这里有一些核心的非线性函数，$\tanh(x), \sigma(x), \text{ReLU}(x)$很常用。你可能会想，为什么是这些函数？我可以想到更多其他的非线性函数。原因就是这些函数的梯度容易计算。对于学习来说，梯度的计算是非常重要的。比如：
$$
\frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))
$$
一个提示：可能你在之前的AI课程上了解到$\sigma(x)$是一个默认的非线性函数。但是人们不常用它，因为它的梯度随着参数绝对值的增长消失的特别快，梯度很小意味着很难学习。所以大部分人使用tanh或者ReLU.
```PYTHON
# In pytorch, most non-linearities are in torch.functional (we have it imported as F)
# Note that non-linearites typically don't have parameters like affine maps do.
# That is, they don't have weights that are updated during training.
data = torch.randn(2, 2)
print(data)
print(F.relu(data))
```
Out:
```PYTHON
tensor([[-0.5404, -2.2102],
        [ 2.1130, -0.0040]])
tensor([[0.0000, 0.0000],
        [2.1130, 0.0000]])
```
### Softmax 和概率
Softmax也是一个非线性函数，但是它经常用在网络的最后一个操作，这是因为它输入一个实数向量，然后返回概率分布。它的定义如下，x是实数的向量，然后Softmax的第i个部分是
$$
\frac{\exp(x_i)}{\sum_j \exp(x_j)}
$$
现在应该明白它的输出是一个概率分布，每一个部分都是非负数，并且所有部分之和为1.
```PYTHON
# Softmax is also in torch.nn.functional
data = torch.randn(5)
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
print(F.log_softmax(data, dim=0))  # theres also log_softmax
```
Out:
```PYTHON
tensor([ 1.3800, -1.3505,  0.3455,  0.5046,  1.8213])
tensor([0.2948, 0.0192, 0.1048, 0.1228, 0.4584])
tensor(1.)
tensor([-1.2214, -3.9519, -2.2560, -2.0969, -0.7801])
```
### 目标函数
目标函数就是你的网络被训练达到最小的函数，也被叫做损失函数。你选择的训练数据通过网络计算，最终计算出输出的损失。网络中的参数通过损失函数的导数进行更新。直观来讲，当模型非常确信自己的结果，但是结果错误时，损失会很大，当模型确信自己的结果并且结果正确时，损失会很小。
最小化损失函数的思想，就是模型希望在未知的测试，开发，生产数据上具有较小的损失。
一个例子就是负对数似然函数损失，这个是在多分类中常用的损失函数，对于监督多分类模型，这意味着训练这个网络最小化正确输出的负对数概率。或者说最大化正确输出的正对数概率。
### 优化和训练
那么我们可以为一个实例计算损失函数了，但是怎么计算呢？在前面我们知道了如何为tensor计算梯度，现在损失函数也是一个tensor,我们可以为我们所有使用过的的参数计算梯度，然后我们可以执行标准的梯度更新。假设$\theta$是我们的参数，$L(\theta)$是损失函数，$\eta$是一个非负的学习速率：
$$
\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta L(\theta)
$$
许多算法和研究热门，在尝试做更多的事情而不是仅仅更新梯度。有许多人在训练时尝试改变学习速率。你不必担心这些算法在做什么，除非你真的对此感兴趣。Torch提供了很多优化算法在torch.optim包里，它们都是透明的。使用最简单的梯度更新和更复杂的梯度更新是一样的。尝试不同的更新算法和不同参数的更新算法在提高模型的的表现是重要的。常常使用优化算法例如Adam 或者 RMSProp代替SGD会显著提升性能。
### 使用pytorch创建网络组件
在我们把注意力转移到NLP之前，我们先做一个有注解的例子，只使用仿射映射和非线性函数。我们也会看到使用负对数似然函数如何计算损失函数，如何通过反向传播更新参数。
所有的网络组件应该继承nn.Module,并且覆盖forward()函数。从nn.Module的继承为组件提供了功能。比如，它记录了自己可以训练的参数。你可以使用.to(device)在cpu和gpu之间交换.device 可以是cpu设备torch.device("cpu")，或者CUDA设备torch.device("cuda:0").
让我们写一个注解的例子，
让我们编写一个带注释的网络示例，该网络采用稀疏的词袋表示，并在两个标签上输出概率分布：“English”和“Spanish”。这个模型只是逻辑回归。
#### 例子：逻辑回归词袋分类器
本分类器使用稀疏的BOW(bag of word)表示方法,输出每一个标签的对数概率。我们为词库中的每个单词指定一个索引。比如：我们词库中只有“hello”和“world”两个单词，用索引0和1表示。“hello hello hello hello”的BOW的向量为[4,0],"hello world world hello"的向量为[2,3].总之也就是对于一个句子，其向量是[count("hello"),count("world")]
假设BOW的向量是x,那么我们网络的输出是
$$
\log \text{Softmax}(Ax + b)
$$
这样，我们把输入通过一个仿射映射，然后做log Softmax.
```PYTHON
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# the model knows its parameters.  The first output below is A, the second is b.
# Whenever you assign a component to a class variable in the __init__ function
# of a module, which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the PyTorch devs, your module
# (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
for param in model.parameters():
    print(param)

# To run the model, pass in a BoW vector
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)
    print(log_probs)
```
Out:
```PYTHON
{'me': 0, 'gusta': 1, 'comer': 2, 'en': 3, 'la': 4, 'cafeteria': 5, 'Give': 6, 'it': 7, 'to': 8, 'No': 9, 'creo': 10, 'que': 11, 'sea': 12, 'una': 13, 'buena': 14, 'idea': 15, 'is': 16, 'not': 17, 'a': 18, 'good': 19, 'get': 20, 'lost': 21, 'at': 22, 'Yo': 23, 'si': 24, 'on': 25}
Parameter containing:
tensor([[ 0.1194,  0.0609, -0.1268,  0.1274,  0.1191,  0.1739, -0.1099, -0.0323,
         -0.0038,  0.0286, -0.1488, -0.1392,  0.1067, -0.0460,  0.0958,  0.0112,
          0.0644,  0.0431,  0.0713,  0.0972, -0.1816,  0.0987, -0.1379, -0.1480,
          0.0119, -0.0334],
        [ 0.1152, -0.1136, -0.1743,  0.1427, -0.0291,  0.1103,  0.0630, -0.1471,
          0.0394,  0.0471, -0.1313, -0.0931,  0.0669,  0.0351, -0.0834, -0.0594,
          0.1796, -0.0363,  0.1106,  0.0849, -0.1268, -0.1668,  0.1882,  0.0102,
          0.1344,  0.0406]], requires_grad=True)
Parameter containing:
tensor([0.0631, 0.1465], requires_grad=True)
tensor([[-0.5378, -0.8771]])
```
上面的对数概率哪一个是和“ENGLISH”对应，哪一个和“SPANISH”对应？我们没有定义它，但是如果我们需要训练就需要定义。
```PYTHON
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}
```
所以让我们训练吧，我们把实例传进去，然后获取到对数概率，计算损失函数，计算损失函数的梯度，使用梯度步长更新参数。在nn包里提供了损失函数
nn.NLLLoss() 是我们需要的负对数似然函数。在torch.optim 中也定义了优化函数。在这里，我们仅仅使用SGD进行优化。
注意：NLLLoss的输入是对数概率的向量和目标标签。它并不会为我们计算对数概率。这就是为什么我们网络的最后一层是log Softmax。
nn.CrossEntropyLoss()和nn.NLLLoss是一样的，只是前者会为你做log Softmax.
```PYTHON
# Run on test data before we train, just to see a before-and-after
with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in range(100):
    for instance, label in data:
        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Tensor as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# Index corresponding to Spanish goes up, English goes down!
print(next(model.parameters())[:, word_to_ix["creo"]])
```
Out:
```PYTHON
tensor([[-0.9297, -0.5020]])
tensor([[-0.6388, -0.7506]])
tensor([-0.1488, -0.1313], grad_fn=<SelectBackward>)
tensor([[-0.2093, -1.6669]])
tensor([[-2.5330, -0.0828]])
tensor([ 0.2803, -0.5605], grad_fn=<SelectBackward>)
```
我们得到了答案，你可以看到在第一个例子中，Spanish的对数概率更高，第二个例子中，English的对数概率更高。
在，您将了解如何创建PyTorch组件，通过它传递一些数据并执行梯度更新。我们准备深入挖掘NLP的深层内容。
## 词嵌入：词汇语义编码
词嵌入是一个实数的稠密向量，在你的词库中，一个词是一个向量。在NLP,大多数情况下你的特征是一个词，但是在计算机中如何表示一个词呢？
你可以存储ascii字符来表示，但是这只是告诉了这个词是什么，但是不能包含它是什么意思。甚至，你能从这些表示中得到什么信息。我们经常需要从神经网络得到稠密的输出，神经网络的输入时|V|维的，V是我们的词汇表。但是经常输出低维，那么我们怎么从高维空间得到低维空间呢？
那么我们使用One-hot 编码代替ascii 编码怎么样？我们可以把w表示为：
$$
\overbrace{\left[ 0, 0, \dots, 1, \dots, 0, 0 \right]}^\text{|V| elements}
$$
其中1的位置是唯一的，其他位置都是0.每个单词的1的位置都是不同的。
这样的话，会有一个很大的缺点，除了这个向量很大之外，每个向量都是独立的，没有联系的。我们真正想要的是单词之间一些相似的概念。为什么？让我们看个例子。
假设我们在建造一个语言模型，假设我们在训练数据中已经看到了下面的句子：
    - The mathematician ran to the store.
    - The physicist ran to the store.
    - The mathematician solved the open problem.
现在假设我们看到了一个之前没有见过的新句子：
    - The physicist solved the open problem.
我们的语言模型可能在这句话上做得很好，但如果我们可以使用以下两个事实就不会好得多：
    - 我们可以看到mathematician和physicist 在句子中具有相同的角色，他们之间有一些语义关系。
    - 正如我们正在看的physicist,mathematician在没有看到的句子中具有相同的角色。
然后我们可以推断，physicist很适合没有看到的句子。这就是我们所说的相似概念：我们所说的语义相似性，不是指具有相似的正交表示，而是通过连接我们看到的和未看到的点，对抗语言数据稀疏性的技术。
这节课程的例子依赖于一个基本的语言假设：如果词语出现在相似的上下文中，那么他们在语义上相互联系。这个叫做分布式假设 （distributional hypothesis）.
### 获取稠密词嵌入
我们怎么解决这个问题？我们如何编码词语的语义相似度？可能我们会想起一些语义的属性。比如，我们看到数学家和物理学家可以跑步，所以我们可以给这些单词的“可以跑步”这个属性给予一个高分。想出其他的属性，然后想象你会给这些常用的单词的这些属性打多少分。
如果每个属性是一个维度，那么我们可以给每个单词像这样的向量表示：
$$
q_\text{mathematician} = \left[ \overbrace{2.3}^\text{can run},
\overbrace{9.4}^\text{likes coffee}, \overbrace{-5.5}^\text{majored in Physics}, \dots \right] 
$$
$$
q_\text{physicist} = \left[ \overbrace{2.5}^\text{can run},
\overbrace{9.1}^\text{likes coffee}, \overbrace{6.4}^\text{majored in Physics}, \dots \right]
$$
那么我们就可以得到两个词的相似程度了
$$
\text{Similarity}(\text{physicist}, \text{mathematician}) = q_\text{physicist} \cdot q_\text{mathematician}
$$
更常用的是把它归一化：
$$
\text{Similarity}(\text{physicist}, \text{mathematician}) = \frac{q_\text{physicist} \cdot q_\text{mathematician}}
{\| q_\text{physicist} \| \| q_\text{mathematician} \|} = \cos (\phi)
$$
其中$\phi$是两个向量的角度，这样的话，如果两个向量很相近，那么相似度接近1，如果两个向量很不相似，那么相似度接近-1。
你可以回想起一开始我们讲的one-hot表示的方法，它可以作为我们这个表示的一种特例：每个单词之间的相似度都是0.每个单词都具有自己的唯一的语义属性。我们现在这个新向量就是稠密向量，里面的实体都不是0.
但是，这个向量有个巨大的痛点：你可以想起成千上万个不同的可能相关的语义属性，然后你怎么设置这些不同属性的值呢？深度学习的中心就是神经网络学习特征的表示，而不是要求程序员自己设计他们。所以为什么不让词嵌入成为模型的参数，然后训练的时候更新他们呢？这就是我们将要去做的。我们将会有一些网络可以学习的潜在语义属性。注意一下，词嵌入可能是不是可以解释的。在上面的手稿中，数学家和物理学家都喜欢咖啡，如果我们让神经网络去学习词嵌入，然后看到数学家和物理学家在一些维度有比较大的值，但是我们不知道这个维度意味着什么。他们在潜在的语义维度具有相似性，但是对于我们是不可解释的。
总之，词嵌入是对词语的语义表示，有效地编码手头上任务相关的语义信息。你也可以嵌入其他的东西：词性标签，解析树，其他东西。特征嵌入的思想是这个领域的核心。
### Pytorch中的词嵌入
在我们开始例子和练习之前，关于如何在Pytorch和深度学习编程中使用嵌入进行一些快速说明。和one-hot中定义唯一的索引很像，当我们使用词嵌入时也需要为每一个词定义一个索引。这些将会是查找表的key,这样，嵌入会保存在|V|xD的矩阵中，D是嵌入的维度，所以索引为i的词，它的嵌入保存在矩阵的第i行。在我的代码中，单词到索引的映射保存在 word_to_idx中。
torch.nn.embedding 模块允许你使用嵌入，它需要两个参数，一个是词汇大小，一个是嵌入的维度。
为了可以索引表中数据，你一定要用torch.LongTensor.
```PYTHON
# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```
```PYTHON
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
```
Out:
```PYTHON
tensor([[ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519]],
       grad_fn=<EmbeddingBackward>)
```
### 一个例子：N-gram 语言模型
回想一下N-gram语言模型，给定一个包含词语w的句子，我们想要计算
$$
P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
$$
$w_i$是这个句子的第i个词语。
在这个例子中，我们将要计算在训练例子上的损失函数，然后使用反向传播更新参数。
```PYTHON
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!
```
Out:
```PYTHON
[(['When', 'forty'], 'winters'), (['forty', 'winters'], 'shall'), (['winters', 'shall'], 'besiege')]
[517.3222274780273, 514.6745493412018, 512.0429074764252, 509.4270143508911, 506.8274371623993, 504.2419238090515, 501.66898941993713, 499.10938024520874, 496.5624313354492, 494.0269808769226]
```
### 练习：计算词嵌入：连续词袋（Continuous Bag-of-Words）
连续词袋（CBOW）在深度学习NLP中经常使用。这个模型试图根据一些单词的上下文来预测单词。这个模型在语言模型中很特别，因为CBOW不是序列的，也不必是概率性的。CBOW用来快速训练词嵌入，这些嵌入用来初始化更加复杂模型的嵌入。通常，这些涉及到预训练嵌入，这个几乎总是可以提高百分之几。
下面就是CBOW模型，假设目标词是$w_i$,在每边有一个N的上下文窗口，$w_{i-1}, \dots, w_{i-N}$和$w_{i+1}, \dots, w_{i+N}$把上下文单词简写为C，CBOW尝试最小化：
$$
-\log p(w_i | C) = -\log \text{Softmax}(A(\sum_{w \in C} q_w) + b)
$$
其中$q_w$是单词$w$的嵌入。
使用pytorch实现这个模型，补全下面的类，这里有一些建议：
    - 思考你有哪些参数需要定义
    - 确定你知道每个操作需要的数据形状，如果需要改变形状，可以使用.view()函数
```PYTHON
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)  # example
```
Out:
```PYTHON
[(['We', 'are', 'to', 'study'], 'about'), (['are', 'about', 'study', 'the'], 'to'), (['about', 'to', 'the', 'idea'], 'study'), (['to', 'study', 'idea', 'of'], 'the'), (['study', 'the', 'of', 'a'], 'idea')]
```
## 序列模型和长短记忆网络
这时候，我们看到了不同的前馈神经网络，但是网络中没有状态保持。所以他们的行为可能不会像我们想的那样。序列模型是NLP的核心。他们是与你输入之间存在某种依赖关系。经典的例子就词性标注的隐马尔可夫模型。另外一个例子就是条件随机场。
一个循环神经网络是可以保持一些状态的网络。比如，它的输出可以作为下个输入的一部分。所以信息可以随着序列在网络中传播。以LSTM为例，对于序列中的每一个元素，有一个对应的隐藏状态$h_i$,可以包含序列中之前点的任意信息。我们在语言模型中会用隐藏状态预测词语，序列标注，和大量其他的事情。
### Pytorch 的LSTM
在开始示例之前，需要注意一些事件。Pytorch的LSTM期望它的输入时的形状是3D tensor.这些tensor的轴的语义是重要的。序列中的第一个轴是它本身，第二个轴是迷你批次中的索引实例，第三个索引是输入的元素。之前我们没有讨论过迷你批次，所以我们忽略它，假设我们总是遇到第二个轴是一维的。如果我们想在序列模型上运行“The cow jumpd”,我们的输入应该像这样：
$$
\begin{split}\begin{bmatrix}
\overbrace{q_\text{The}}^\text{row vector} \\\\
q_\text{cow} \\\\
q_\text{jumped}
\end{bmatrix}\end{split}
$$
除了记住这是一个附加的大小为1的第二维。
此外，您可以一次查看一个序列，在这种情况下，第一个轴也将具有大小1。
让我们看一个简单的例子。
```PYHON
# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```
```PYTHON
lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
```
Out:
```PYTHON
tensor([[[-0.0187,  0.1713, -0.2944]],

        [[-0.3521,  0.1026, -0.2971]],

        [[-0.3191,  0.0781, -0.1957]],

        [[-0.1634,  0.0941, -0.1637]],

        [[-0.3368,  0.0959, -0.0538]]], grad_fn=<StackBackward>)
(tensor([[[-0.3368,  0.0959, -0.0538]]], grad_fn=<StackBackward>), tensor([[[-0.9825,  0.4715, -0.0633]]], grad_fn=<StackBackward>))
```
### 例子：LSTM进行词性标注
在这个例子中，我们将使用LSTM进行序列标注。我们不使用类似维比特算法和前向后向算法的算法。作为一个练习，在看到下面的算法后，你可以考虑在这之中如何使用维比特算法。
这个模型是这样的，我们的输入时$w_1,w_2,...w_M$,其中$w_i \in V$,V是我们的词汇，T是我们的标注，$y_i$是我们对单词$w_i$的标注。我们用$\hat{y_i}$表示我们对$w_i$预测的标签。
这是一个结构化的预测模型，我们的输出是一个序列$\hat{y_1},\dots,\hat{y_M}$，其中$\hat{y_i} \in T$
为了做这个预测，在序列上使用LSTM,假设在时间i时，隐藏状态是$h_i$.假设每个标签是不同的索引（像我们做词嵌入时，word_to_idx那样）。然后我们对$\hat{y_i}$的预测规则是
$$
\hat{y}_i = \text{argmax}_j \\  (\log \text{Softmax}(Ah_i + b))_j
$$
其实就是对隐藏状态做仿射映射，然后做log softmax 。预测的结果就是向量里面的最大的值。这就意味着A的目标空间维度是|T|。
```PYTHON
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
```
Out:
```PYTHON
{'The': 0, 'dog': 1, 'ate': 2, 'the': 3, 'apple': 4, 'Everybody': 5, 'read': 6, 'that': 7, 'book': 8}
```
Create the model:
```PYTHON
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
```
Train the model:
```PYTHON
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
```
OUT:
```PYTHON
tensor([[-1.1389, -1.2024, -0.9693],
        [-1.1065, -1.2200, -0.9834],
        [-1.1286, -1.2093, -0.9726],
        [-1.1190, -1.1960, -0.9916],
        [-1.0137, -1.2642, -1.0366]])
tensor([[-0.0462, -4.0106, -3.6096],
        [-4.8205, -0.0286, -3.9045],
        [-3.7876, -4.1355, -0.0394],
        [-0.0185, -4.7874, -4.6013],
        [-5.7881, -0.0186, -4.1778]])
```
### 练习：使用字符特征扩展LSTM词性标注
在上面的例子中，每个单词都有嵌入作为我们模型的输入，让我们使用单词派生的方法扩展词嵌入。我们预计这个会有很大的帮助。由于单词基本的信息，像词缀，在词性标记中有很大的作用。比如在英语中，对于词缀-ly经常被标记为副词。
为了做这些，让$c_w$表示单词w的字符级表示。让$x_w$表示之前的词嵌入。然后我们的输入是$x_w$和$c_w$的连接。所以如果$x_w$的维度是5，$c_w$的维度是3，那么LSTM的应该接受的输入维度是8。
为了得到字符级表示，使用单词的字符作LSTM,让$c_w$成为LSTM的最终隐藏状态。有一些提示：
    -在新的模型中应该有两个LSTM,原始的那个输出POS标记的分数，新的输出每个单词的字符级表示。
    -为了在字符上做序列模型，你应该对字符做嵌入。字符的嵌入将会输入到字符LSTM中。
## 进阶：动态决定和BI-LSTM CRF
### 动态和静态深度学习工具包
Pytorch 是一个动态深度学习包。另一个动态深度学习包的例子是Dynet(我提到这个是因为Dynet和Pytorch比较相似，如果你看到了Dynet写的例子，会帮助你使用pytorch实现它)。相对应的是静态工具包，包含Theano, Keras, TensorFlow等等。他们之间最核心的不同是：
    - 在静态工具包中，你定义了计算图一次，然后编译它，然后实例流式传给它。
    - 在动态工具包中，你会为每个实例定义计算图，它不会被编译，它会即时执行的。
没有很多经验的话，很难领会到差异，一个例子是我们想建立一个深层成分分析器，假设我们的模型大概包含下面的步骤：
    - 我们从下向上建立树
    - 标记根节点（序列的单词）
    - 然后，使用神经网络和单词的嵌入发现成分的组合，无论何时你组成一个成分时，使用一些技术获得成分的嵌入。这样，我们的网络完全依赖输入序列。在这个句子“The green cat scratched the wall”，在这个模型的某个地方，我们想组合这个范围$(i,j,r) = (1, 3, \text{NP})$(意思是一个NP成分从第1个单词到第3个单词，也就是“The green cat”)
然而，另一个句子可能是“Somewhere, the big fat cat scratched the wall”。在这个句子中，我们在某个地方想要组成这个成分$(2, 4, NP)$。这个成分将会依赖这个实例。如果我们在静态工具包里编译这个计算图一次，这将会非常困难或者根本不可能。在动态工具包里，这就仅仅是一个预定义的计算图。它可以是为每一个实例生成新计算图。所以就没有这个问题了。
动态工具包有一些优点，比如更容易debug,更类似于宿主语言（意思是，Pytorch和Dynet比Keras和Theano 更像Python）。
### BI-LSTM 条件随机场讨论
在这个部分，我们将会看到一个完整，复杂的BI-LSTM 条件随机场的例子。之前的LSTM标记只是满足了词性标注，但是一个像CRF这样的序列模型对于NER表现很好。假设我们已经熟悉CRF,虽然明着听着可怕，但是所有模型都是CRF,但是LSTM提供了一些功能。这是一个先进的模型，比我们这个教程的其他模型都要复杂，如果你想跳过这个也没有关系。如果你准备好了，看看你是否可以：
    - 在步骤i中为标记k写出维特比变量的循环。
    - 修改上面的循环计算前向变量
    - 修改上面的循环在log空间计算前向变量
如果你可以做上面的三件事，那么你应该可以理解下面的代码。回想CRF计算了条件概率。假设y是标签序列，x是输入序列。然后我们计算
$$
P(y|x) = \frac{\exp{(\text{Score}(x, y)})}{\sum_{y'} \exp{(\text{Score}(x, y')})}
$$
其中的sorce由定义一些对数势$\log \psi_i(x,y)$来定义：
$$
\text{Score}(x,y) = \sum_i \log \psi_i(x,y)
$$
为了让分隔函数易于管理，势必须只看本地的特征。
在BI-LSTM CRF 中，我们定义了两种势，发射和转移，在索引i出单词的势来自于在时间点i时，Bi-LSTM的隐藏状态。这个转移分数存储在|T|x|T|的矩阵$\textbf{P}$中,T是标签集。在我的实现中，$\textbf{P}_{j,k}$是从标签k到标签j的转移分数。所以：
$$
\text{Score}(x,y) = \sum_i \log \psi_\text{EMIT}(y_i \rightarrow x_i) + \log \psi_\text{TRANS}(y_{i-1} \rightarrow y_i)  \\\\
= \sum_i h_i[y_i] + \textbf{P}_{y_i, y_{i-1}}
$$
在第二个式子中，我们认为标签被用非负的索引标记。
如果上面讨论太简略，你可以看一下Michael Collins写的关于CRF的[文章](http://www.cs.columbia.edu/~mcollins/crf.pdf)
### 实现笔记
```PYTHON
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
```
帮助函数可以让代码更易读
```PYTHON
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
```
创建模型
```PYTHON
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
```
进行训练：
```PYTHON
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!
```
输出：
```PYTHON
(tensor(2.6907), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1])
(tensor(20.4906), [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
```
### 练习：辨别标记的新损失函数
在做解码的时候，我们不必创建一个计算图，因为我们不会从维比特路径分反向传播。因为我们无论如何都有它，尝试训练这个标记器，其中损失函数是维比特路径得分和金标准路径之间的差异，应该清楚的是，当预测是序列是正确的序列时，这个函数是非负的和0，这个基本上是结构感知器。
这个改动应该是比较短的，因为维比特算法和分数序列已经实现了。这是依赖训练实例的计算图的例子，虽然我没有使用静态工具包实现这个，但我能想象的到，那是可能的但是不会那么简单。
找一些实际的数据做一下对比吧！
