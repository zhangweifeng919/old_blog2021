---
title: 结巴（jieba）分词源码分析
date: 2019-05-07 10:17:26
tags:
- nlp
- jieba
categories:
- 源码阅读笔记
description: 结巴分词是中文分词中最常用的分词组件
---

# jieba

“结巴”中文分词：做最好的 Python 中文分词组件

# 特点

 1. 支持三种分词模式：
  1. 精确模式，试图将句子最精确地切开，适合文本分析；
  2. 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
  3. 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
 2. 支持繁体分词
 3. 支持自定义词典
 4. MIT 授权协议

# 安装说明
代码对 Python 2/3 均兼容
 1. 全自动安装：easy_install jieba 或者 pip install jieba / pip3 install jieba
 2. 半自动安装：先下载 http://pypi.python.org/pypi/jieba/ ，解压后运行 python setup.py install
 3. 手动安装：将 jieba 目录放置于当前目录或者 site-packages 目录
 4. 通过 import jieba 来引用 

# 算法
 1. 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图 (DAG)
 2. 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
 3. 对于未登录词，采用了基于汉字成词能力的 HMM 模型，使用了 Viterbi 算法

# 主要功能
 # 分词
 1. jieba.cut 方法接受三个输入参数: 需要分词的字符串；cut_all 参数用来控制是否采用全模式；HMM 参数用来控制是否使用 HMM 模型
 2. jieba.cut_for_search 方法接受两个参数：需要分词的字符串；是否使用 HMM 模型。该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细
 3. 待分词的字符串可以是 unicode 或 UTF-8 字符串、GBK 字符串。注意：不建议直接输入 GBK 字符串，可能无法预料地错误解码成 UTF-8
 4. jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)，或者用
 5. jieba.lcut 以及 jieba.lcut_for_search 直接返回 list
 6. jieba.Tokenizer(dictionary=DEFAULT_DICT) 新建自定义分词器，可用于同时使用不同词典。jieba.dt 为默认分词器，所有全局分词相关函数都是该分词器的映射。
 
 # 实现
  1. 生成前缀字典
 ```python
     def gen_pfdict(self, f):
        lfreq = {} #前缀字典，key是词，value 是频率
        ltotal = 0 # 词的频率总数
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode('utf-8')
                word, freq = line.split(' ')[:2]
                freq = int(freq)
                lfreq[word] = freq
                ltotal += freq
                for ch in xrange(len(word)):
                    wfrag = word[:ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        f.close()
        return lfreq, ltotal


 ```
  2. 获取DAG图
 ```python
     def get_DAG(self, sentence):
        self.check_initialized() # 检查初始化
        DAG = {} # key是字的位置，value是一个list,比如{1:[1,2,3]}表示坐标1到1，1到2，1到3，都可以组成一个词
        N = len(sentence)
        for k in xrange(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

 ```
  3. 通过上面两小节可以得知，我们已经有了词库(dict.txt)的前缀字典和待分词句子sentence的DAG，基于词频的最大切分 要在所有的路径中找出一条概率得分最大的路径，该怎么做呢？ 
jieba中的思路就是使用动态规划方法，从后往前遍历，选择一个频度得分最大的一个切分组合。 
```python
#动态规划，计算最大概率的切分组合
    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
         # 对概率值取对数之后的结果(可以让概率相乘的计算变成对数相加,防止相乘造成下溢)
        logtotal = log(self.total)
        # 从后往前遍历句子 反向计算最大概率
        for idx in xrange(N - 1, -1, -1):
           # 列表推倒求最大概率对数路径
           # route[idx] = max([ (概率对数，词语末字位置) for x in DAG[idx] ])
           # 以idx:(概率对数最大值，词语末字位置)键值对形式保存在route中
           # route[x+1][0] 表示 词路径[x+1,N-1]的最大概率对数,
           # [x+1][0]即表示取句子x+1位置对应元组(概率对数，词语末字位置)的概率对数
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])
```
从代码中可以看出calc是一个自底向上的动态规划(重叠子问题、最优子结构)，它从sentence的最后一个字(N-1)开始倒序遍历sentence的字(idx)的方式，计算子句sentence[idx~N-1]概率对数得分（这里利用DAG及历史计算结果route实现，同时赞下 作者的概率使用概率对数 这样有效防止 下溢问题）。然后将概率对数得分最高的情况以（概率对数，词语最后一个字的位置）这样的tuple保存在route中。 
  4. 获取切分结果
```python
 def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.get_DAG(sentence) # 获取dag
        route = {}
        self.calc(sentence, DAG, route) # 获取最大概率切分组合
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y] # 以x为起点的最大概率切分词语
            if re_eng.match(l_word) and len(l_word) == 1: # 数字或者字母
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
```
  5. 测试
```python
#coding:utf8
'''
 测试jieba 文件
'''
import jieba
from jieba import Tokenizer
s="去北京大学玩"
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
print('/'.join(cut_DAG_NO_HMM(sentence=s,DAG=dag,route=myroute)))
```
  6. 结果
![分词测试NO_HMM](/images/源码阅读笔记/结巴/分词测试NO_HMM.png)
# 参考资料
[jieba中文分词源码分析（三）](https://blog.csdn.net/daniel_ustc/article/details/48223135) 