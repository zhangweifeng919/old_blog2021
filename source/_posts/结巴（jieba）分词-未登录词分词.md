---
title: 结巴（jieba）分词-未登录词分词
date: 2019-05-08 15:03:46
tags:
- nlp
- jieba
categories:
- 源码阅读笔记
description: 对于未登录词，jieba 采用了HMM模型进行解决
---


## HMM 分词原理

具体的详细介绍可以参考网上的资料，或者我之前写的读书笔记[统计自然语言处理-概率图模型](/2019/04/15/统计自然语言处理-概率图模型/)，这里只说HMM在分词中主要解决三个问题。
1. 评估问题(概率计算问题) 
即给定观测序列 O=O1,O2,O3…Ot和模型参数λ=(A,B,π)，怎样有效计算这一观测序列出现的概率. 
(Forward-backward算法) 
2. 解码问题(预测问题) 
即给定观测序列 O=O1,O2,O3…Ot和模型参数λ=(A,B,π)，怎样寻找满足这种观察序列意义上最优的隐含状态序列S。 
(viterbi算法,近似算法) 
3. 学习问题 
即HMM的模型参数λ=(A,B,π)未知，如何求出这3个参数以使观测序列O=O1,O2,O3…Ot的概率尽可能的大. 
(即用极大似然估计的方法估计参数,Baum-Welch,EM算法)

HMM 模型的五元组表示： 
```JAVA
{ 
states，//状态空间 
observations，//观察空间 
start_probability，//状态的初始分布，即π 
transition_probability，//状态的转移概率矩阵，即A 
emission_probability//状态产生观察的概率，发射概率矩阵,即B 
}
```
## HMM对未登录词的识别
先看一个测试
```python
#coding:utf8
'''
 测试jieba 文件
'''
import jieba
from jieba import Tokenizer
s="到MI京研大厦"
t=Tokenizer()
t.initialize()
dag=t.get_DAG(s)
print(dag)
for i in range(len(s)):
    for j in dag[i]:
            print(i,j,s[i:j+1])
myroute={}
t.calc(s,dag,myroute)
print(myroute)


def cut_DAG_NO_HMM(sentence,DAG,route):
    x = 0
    N = len(sentence)
    buf = ''
    while x < N:
        y = route[x][1] + 1
        l_word = sentence[x:y] # 以x为起点的最大概率切分词语
        if jieba.re_eng.match(l_word) and len(l_word) == 1: # 数字或者字母
            buf += l_word
            x = y
        else:
            if buf:
                yield buf
                buf = ''
            yield l_word
            x = y
    if buf:
        yield buf
        buf = ''
def cut_DAG(sentence,DAG,route,tokenizer):
    x = 0
    buf = ''
    N = len(sentence)
    while x < N:
        y = route[x][1] + 1
        l_word = sentence[x:y]
        if y - x == 1:
            buf += l_word
        else:
            if buf:
                if len(buf) == 1:
                    yield buf
                    buf = ''
                else:
                    if not tokenizer.FREQ.get(buf):
                        recognized = jieba.finalseg.cut(buf)
                        for t in recognized:
                            yield t
                    else:
                        for elem in buf:
                            yield elem
                    buf = ''
            yield l_word
        x = y

    if buf:
        if len(buf) == 1:
            yield buf
        elif not t.FREQ.get(buf):
            recognized = jieba.finalseg.cut(buf)
            for t in recognized:
                yield t
        else:
            for elem in buf:
                yield elem



print('/'.join(cut_DAG_NO_HMM(sentence=s,DAG=dag,route=myroute)))
print('/'.join(cut_DAG(sentence=s,DAG=dag,route=myroute,tokenizer=t)))
```
显示的结果
![分词测试WITH_HMM](/images/源码阅读笔记/结巴/分词测试WITH_HMM.png)
我们发现“京研”这个词被正确的识别出来了，这就是HMM的强大之处。这是HMM的三大问题之一，由观察序列，求隐含序列。
上一节中我们说明了HMM由五元组表示，那么这样的五元组参数在中文分词中的具体含义是：
1. states & observations 状态空间和观察空间
    汉字按照BEMS四个状态来标记，分别代表 Begin End Middle 和 Single， {B:begin, M:middle, E:end, S:single}。分别代表每个状态代表的是该字在词语中的位置，B代表该字是词语中的起始字，M代表是词语中的中间字，E代表是词语中的结束字，S则代表是单字成词。 
    观察空间为就是所有汉字(我她…)，甚至包括标点符号所组成的集合。 
    状态值也就是我们要求的值，在HMM模型中文分词中，我们的输入是一个句子(也就是观察值序列)，输出是这个句子中每个字的状态值，用这四个状态符号依次标记输入句子中的字，可方便的得到分词方案。 如： 
    观察序列：我在北京 
    状态序列：SSBE 
    对于上面的状态序列，根据规则进行划分得到 S/S/BE/ 
    对应于观察序列：我/在/北京/ 
    分词任务就完成了。 
    同时我们可以注意到： 
    B后面只可能接(M or E)，不可能接(B or E)。而M后面也只可能接(M or E)，不可能接(B, S)。
2. start_probability 状态初始分布 
    这个很好理解，如下：
    ```PYTHON
    P={ 'B': -0.26268660809250016,
        'E': -3.14e+100,
        'M': -3.14e+100,
        'S': -1.4652633398537678}
    ```
    示例数值是对概率值取对数之后的结果(s让概率相乘变成对数相加)，其中-3.14e+100作为负无穷，也就是对应的概率值是0。它表示了一个句子的第一个字属于{B,E,M,S}这四种状态的概率，如上可以看出，E和M的概率都是0，这和实际相符合，开头的第一个字只可能是词语的首字(B)，或者是单字成词(S)，这部分内容对应 jieba/finalseg/prob_start.py文件，具体源码。
3. transition_probability（状态的转移概率矩阵） 
转移概率是马尔科夫链很重要的一个知识点，马尔科夫链(一阶)最大的特点就是当前T=i时刻的状态state(i)，只和T=i时刻之前的n个状态有关，即: 
{state(i-1), state(i-2), … state(i - n)} 
HMM模型有三个基本假设： 
a. 系统在时刻t的状态只与时刻t-1处的状态相关,(也称为无后效性）; 
b. 状态转移概率与时间无关,(也称为齐次性或时齐性); 
c. 假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态，与其它观测及状态无关,(也称观测独立性假设)。 
其中前两个假设为马尔科夫模型的假设。 模型的这几个假设能大大简化问题。 
再看下transition_probability，其实就是一个嵌套的字典，数值是概率求对数后的值,示例: 
```PYTHON
P={'B': {'E': -0.510825623765990, 'M': -0.916290731874155}, 
'E': {'B': -0.5897149736854513, 'S': -0.8085250474669937}, 
'M': {'E': -0.33344856811948514, 'M': -1.2603623820268226}, 
'S': {'B': -0.7211965654669841, 'S': -0.6658631448798212}} 
```
如P[‘B’][‘E’]代表的含义就是从状态B转移到状态E的概率，由P[‘B’][‘E’] = -0.510825623765990，表示状态B的下一个状态是E的概率对数是-0.510825623765990。 
这部分内容对应 jieba/finalseg/prob_trans.py文件
4. emission_probability(状态产生观察的概率，发射概率) 
根据HMM观测独立性假设发射概率，即观察值只取决于当前状态值，也就是: 
P(observed[i], states[j]) = P(states[j]) * P(observed[i]|states[j]),其中P(observed[i]|states[j])这个值就是从emission_probability中获取。 
emission_probability示例如下：
```PYTHON 
P={'B': {'\u4e00': -3.6544978750449433, 
'\u4e01': -8.125041941842026, 
'\u4e03': -7.817392401429855, 
'\u4e07': -6.3096425804013165, 
..., 
'S':{...}, 
... 
}
```
比如P[‘B’][‘\u4e00’]代表的含义就是’B’状态下观测的字为’\u4e00’(对应的汉字为’一’)的概率对数P[‘B’][‘\u4e00’] = -3.6544978750449433。 
## 维比特算法

到这里已经结合HMM模型把jieba的五元参数介绍完，这五元的关系是通过一个叫Viterbi的算法串接起来，observations序列值是Viterbi的输入，而states序列值是Viterbi的输出，输入和输出之间Viterbi算法还需要借助三个模型参数，分别是start_probability，transition_probability，emission_probability。对于未登录词（OOV）的问题，即已知观察序列S，初始状态概率prob_start，状态观察发射概率prob_emit，状态转换概率prob_trans。 求状态序列W，这是个解码问题，维特比算法可以解决。

该算法可以参考网络资料或者我之前写的读书笔记[统计自然语言处理-概率图模型](/2019/04/15/统计自然语言处理-概率图模型/)

## jieba 分词中的应用
jieba中对于未登录词问题，通过__cut_DAG 函数我们可以看出这个函数前半部分用 calc 函数计算出了初步的分词，而后半部分就是就是针对上面例子中未出现在语料库的词语进行分词了。 
由于基于频度打分的分词会倾向于把不能识别的词组一个字一个字地切割开，所以对这些字的合并就是识别OOV的一个方向，__cut_DAG定义了一个buf 变量收集了这些连续的单个字，最后把它们组合成字符串再交由 finalseg.cut 函数来进行下一步分词。

```PYTHON
# 利用 viterbi算法得到句子分词的生成器
def __cut(sentence):
    global emit_P
    # viterbi算法得到sentence 的切分
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield sentence[begin:i + 1]
            nexti = i + 1
        elif pos == 'S':
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]
```
对应的viterbi算法
```PYTHON
#状态转移矩阵，比如B状态前只可能是E或S状态  
PrevStatus = {  
    'B':('E','S'),  
    'M':('M','B'),  
    'S':('S','E'),  
    'E':('B','M')  
}  
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # 状态概率矩阵  
    path = {}
    for y in states:  # 初始化状态概率
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y] # 记录路径
    for t in xrange(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            # t时刻状态为y的最大概率(从t-1时刻中选择到达时刻t且状态为y的状态y0)
            (prob, state) = max([(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in PrevStatus[y]])
            V[t][y] = prob
            newpath[y] = path[state] + [y] # 只保存概率最大的一种路径 
        path = newpath 
    # 求出最后一个字哪一种状态的对应概率最大，最后一个字只可能是两种情况：E(结尾)和S(独立词)  
    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES')
```
其实到这里思路很明确了，给定训练好的模型(如HMM)参数(λ=(A,B,π)), 然后对模型进行载入，再运行一遍Viterbi算法，就可以找出每个字对应的状态（B, M, E, S），这样就可以根据状态也就可以对句子进行分词。

## 参考资料
[jieba中文分词源码分析（四）](https://blog.csdn.net/daniel_ustc/article/details/48248393) 