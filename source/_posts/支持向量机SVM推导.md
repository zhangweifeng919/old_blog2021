---
title: 支持向量机SVM推导
date: 2019-04-08 10:21:10
tags:
- 机器学习
categories:
- 读书笔记 
description: 介绍SVM的公式推导
---

## 间隔和支持向量

在样本空间中，划分超平面可通过如下线性方程来描述：
$$ \mathbf{w}^T x+ b=0$$
其中$\mathbf{w}=(w_1;w_2;...;w_d)$为法向量，决定了超平面的方向；b为位偏移，决定了超平面与原点之间的距离。
显然，划分超平面可以被法向量$\mathbf{w}$和位移b确定，我们将其记为$(\mathbf{w},b)$。样本空间中任意点$\mathbf{x}$到超平面的距离可以写为
$$
\tau = \cfrac{|\mathbf{w}^T \mathbf{x} + b|}{|| \mathbf{w} ||}
$$
假设超平面$(\mathbf{w},b)$能将训练样本正确分类，即对于$(\mathbf{x}_i,y_i)$,若$y_i=+1$,则有$\mathbf{w}^T + b>0$;
若$y_i=-1$,则有$\mathbf{w}^T + b<0$;若超平面$(\mathbf{w}^,,b^,)$能将训练样本正确分类，则总存在缩放变换$\zeta\mathbf{w} \to \mathbf{w}' ,\zeta b \to b'$使得下式成立:
$$
\begin{cases}
    \mathbf{w}^T\mathbf{x}_i + b \ge +1,&  & y_i=+1; \\\\
    \mathbf{w}^T\mathbf{x}_i + b \le -1,&  & y_i=-1;
\end{cases}
$$
距离超平面最近的几个训练样本点使得上式成立，它们被称为”支持向量“（support vector），两个异类支持向量到超平面的距离之和为：
$$
\gamma=\cfrac{2}{||\mathbf{w}||}
$$
它被称为”间隔“。
欲找到最大间隔的划分超平面，也就是找到参数$\mathbf{w}$和b，使得$\gamma$最大，即：
$$
\begin{align}
& \max_{\mathbf{w},b} \cfrac{2}{||\mathbf{w}||}  \\\\
& s.t. \quad y_i (\mathbf{w}^T \mathbf{x_i} + b) \ge 1,\quad i=1,2,...,m.
\end{align} \qquad (6.5)
$$
显然，为了最大化间隔，仅需最大化$\cfrac{1}{||\mathbf{w}||}$,等价最小化$||\mathbf{w}||^2$,于是可以重写为：
$$
\begin{align}
 & \min_{\mathbf{w},b} \cfrac{1}{2}{||\mathbf{w}||}^2  \\\\
 & s.t. \quad y_i (\mathbf{w}^T\mathbf{x}_i + b) \ge 1,\quad i=1,2,...,m. 
 \end{align} \qquad (6.6)
$$
间隔貌似仅与$\mathbf{w}$相关，其实b通过约束隐式的影响着w的取值，进而对间隔产生影响。
这就是支持向量机的基本型

## 对偶问题

我们要求解（6.6）来得到最大间隔划分超平面对应的模型
$$
f(x)=\mathbf{w}^{T}\mathbf{x} +b \qquad (6.7)
$$
其中$\mathbf{w}$和b是模型参数,注意到（6.6）是一个凸优化问题，可以使用现成的优化包求解，但是我们又更加高效的办法。
对式（6.6）使用拉格朗日乘子法可以得到其对偶问题，具体来说是对（6.6）的每条约束添加拉格朗日乘子$\alpha_i \ge 0$,则该问题拉格朗日函数可以写为
$$
L(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\|\boldsymbol{w}\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right) \quad （6.8）
$$
其中，$\boldsymbol{\alpha} = (\alpha_1;\alpha_2; \cdots;\alpha_m)$.令$L(\boldsymbol{w}, b, \boldsymbol{\alpha})$对$\boldsymbol{w}$和b的偏导为0可以得：
![svm对w和b求偏导](/images/读书笔记/机器学习西瓜书/svm对w和b求偏导.png)
将(6.9)带入(6.8),消除$\mathbf{w}$和b，再考虑消除（6.10）的约束就得到了（6.6）的对偶问题
$$
\underset{\alpha}{max} \sum_{i=1}^m \alpha_i - \cfrac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \\\\
s.t \quad \underset{i=1}{\sum}^m \alpha_i y_i=0, \\\\
\alpha_i \ge 0,i=1,2,\cdots,m  \qquad (6.11)
$$
![6.11推导](/images/读书笔记/机器学习西瓜书/6.11推导.png)
求解出$\alpha$后，求出$\mathbf{w}$和b即可得到模型
$$
\begin{align}
f(x) & =\mathbf{w}^T\mathbf{x} +b  \\\\
&= \sum_{i=1}^m \alpha_i y_i \mathbf{x}_i^T \mathbf{x} + b \quad (6.12)
\end{align}
$$
从对偶问题（6.11）解出来的$\alpha_i$是式（6.8）中的拉格朗日乘子，它对应着样本$(\mathbf{x}_i,y_i)$,注意（6.6）含有不等式约束，因此上述过程需要满足KKT条件
即：
$$
\begin{cases}
\alpha_i \ge 0 \\\\
y_i f(\mathbf{x}_i)-1 \ge 0 \\\\
\alpha_i (y_i f(\mathbf{x}_i)-1)=0
\end{cases} \quad (6-13)
$$
于是，对于任意训练样本$(\mathbf{x}_i,y_i)$,总有$\alpha_i$=0,或者$y_i f(\mathbf{x}_i)=1$，若$\alpha_i=0$,则这样本不会在式（6.12）的求和中出现，也不会对$f(\mathbf{x})$有任何影响；若$\alpha_i \gt 0$则必有$y_i f(\mathbf{x}_i)=1$，所对应的样本点在最大间隔边界上，是一个支持向量。这显示了支持向量机的一个重要性质：训练完成后，大部分的训练样本不需要保存，最终模型仅与支持向量有关。
那么如何求解（6.11）呢？不难发现，这是一个二次规划问题，可使用通用的二次规划算法求解，然而该问题的规模正比于训练样本数，这会在实际任务中造成很大的开销，为了避免这个障碍，人们通过利用问题本身的特性，提出了很多高效的算法，SMO是其中一个著名的代表。
SMO的基本思路是先固定$\alpha_i$之外的所有参数，然后求$\alpha_i$上的极值。由于存在约束$\sum_{i=1}^m \alpha_i y_i =0$,若固定$\alpha_i$之外的其他变量，则$\alpha_i$可以由其他变量导出。于是，SMO每次选择两个变量$\alpha_i$和$\alpha_j$并固定其他参数，这样，在参数初始化后，SMO不断执行如下两个步骤直至收敛：
1. 选取一对需要更新的变量$\alpha_i$和$\alpha_j$
2. 固定$\alpha_i$和$\alpha_j$之外的所有参数,求解（6.11）获得更新后的$\alpha_i$和$\alpha_j$
![SVM的求解](/images/读书笔记/机器学习西瓜书/SVM的求解.png)

## 核函数

在之前的假设中，我们假设了样本数据是线性可分的，即存在一个超平面可以将样本分类。然而现实任务中，原始样本空间内也许并不存在一个能正确划分两类空间的超平面，这样我们可以把样本映射到更高的特征空间中，使得样本在这个特征空间内线性可分。
令$\phi(\boldsymbol{x})$表示$\boldsymbol{x}$映射后的特征向量，于是在特征空间中划分超平面所对应的模型表示为
$$
f(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}} \phi(\boldsymbol{x})+b
$$
其中$\boldsymbol{w}$和b 是模型参数，类似(6.6),有：
$$
\begin{array}{l}{\min_{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}} \\\\ {\text { s.t. } \quad y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \phi\left(\boldsymbol{x}_{i}\right)+b\right) \geqslant 1, \quad i=1,2, \ldots, m}\end{array}
$$
其对偶问题是：
![核函数1](/images/读书笔记/机器学习西瓜书/核函数1.png)
![核函数2](/images/读书笔记/机器学习西瓜书/核函数2.png)



## 参考资料
《机器学习》 周志华 著  清华大学出版社 [机器学习_周志华.pdf下载](/books/pdf/机器学习_周志华.pdf)
《机器学习公式推导-南瓜书》 [南瓜书](https://datawhalechina.github.io/pumpkin-book/#/chapter6/chapter6?id=_69-610)

