---
title: >-
  Methods for Segmentation and Classification of Digital Microscopy Tissue
  Images
date: 2019-09-05 22:24:47
tags:
- 医学图像处理
- 图像处理
categories:
- 论文阅读笔记
description: 数字显微组织图像的分割和分类方法，原始论文：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6454006/
---

## 摘要
组织图片的分析可以帮助癌症研究者更好的理解癌症生物学。
在组织图片分析中，对细胞核的分离和组织图片的分类是普遍的任务。
由于组织形态的复杂性和肿瘤的不稳定性，想要开发出准确和高效的算法成为一个挑战的问题。
开发了两个算法：
一个是实现了多尺度深度残差聚合网络，对细胞核材料进行准确的分割，然后把成群的细胞核分割成独立的细胞核
另一个是分类算法，该算法最初通过深度学习的方法进行补丁级（patch-level）分类。补丁级的统计和形态特征被用来一个随机森林模型的输入。这个随机森林模型是就是整个载片图像的分类的模型。
## 介绍
在过去的20年里，载片图像技术获得了巨大的发展，可以很容易的获取到超高分辨率的图像。
对于使用可靠和高效的计算机方法来补充人工的器官组织检查的需求也越来越高。
在整个载片组织图像分析中，最重要的两个任务就是：肿瘤区域和非肿瘤区域微观结构（细胞核和细胞）的分割，以及图像区域和整个图像的分类
虽然很多人在分割和分类上做了很多研究工作，但是从数字载片图像中提取，挖掘和解释里面信息的过程仍然有很多困难。
这里仍然有很多挑战需要分割和分类算法去解决：
肿瘤和正常组织的形态在不同的组织样本下是不同的--不管是不同癌症类型还是同一个癌症类型的不同样本。即使是同一个样本，也会包含不同种类的细胞核和组织结构。
组织图像中的细胞核会挨着或者重叠在一起。
整个载片的图像具有非常高的分辨率，因此很可能放入大多数的计算机主存和GPU内存中。
在这个文章中，我们展示和通过实验评估了两个新算法，一个是为了细胞核的分割，一个是为了对整个载片组织图像的分类。

分割算法为了提高分割的准确度，提出了一个多尺度深度残差聚合网络（ multiscale deep residual aggregation network ）这个算法包含3个步骤：
首先通过一组CNN探测细胞核的斑点和边界。然后通过分水岭算法对细胞群进行初始的分割，最后一步是对上一步分开的的细胞核进行精细分割。
这个提出的方法中，因为样本细胞核的大小不同，所以采取了多尺度的方法以改善检测和分割的性能。
使用了来自四种病例的图像块评估这个算法，能够得到0.78的准确度。

分类算法提出了两段式的自动化方法来解决非小细胞肺癌病理图像的分类挑战。
首先把从未知的WSI（whole slide image）中获取的输入补丁（input patches），分为LUAD （NSCLC adeno ），LUSC (squamous cell  ) 和   ND
（non-diagnostic）三类。然后获取到每个类别的概率图，然后从LUAD和LUSC中获取到统计上的和形态上的特征集合，作为随机森林回归模型的输入，
对每个WSI进行分类。该方法是第一个为了将每个WSI分类为诊断和未诊断区域的3类网络，实验结果表明该方法具有0.81的准确度。

## 材料和方法
### 用深度学习的方法分割细胞核
开发了一个精确的分割细胞核的卷积神经网络方法。包含了三个主要步骤：
使用CNNs检测细胞核的斑点（blob）和边界。
使用分水岭算法把上步检测的结果结合起来，分割粘连或者重叠的细胞核。
最后是对个别的细胞核的分割
![图一](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g0001.jpg)
Overview of the nuclei segmentation procedure. $DRAN_{BL}$ and $DRAN_{BD}$ are the models for nuclei blob detection and boundary detection, respectively.
#### 深度残差聚合网络 
训练了两个CNN，探测细胞核的斑点和细胞的边界，这些CNN包含了两个连续的处理路径---收缩路径和膨胀路径
目的是获取到组织图像上所有的细胞核像素和细胞核边界像素，生成细胞核斑点和边界遮罩。获取遮罩之后，就可以进行初始的分割。分割分为两步：1.是细胞核边界移除，2.剩下细胞核群的分割。为了移除细胞核边界，使用了一个大小为3x3的核来膨胀边界遮罩，之后，再从团块遮罩上去除边界遮罩。然后使用分水岭算法，识别出独立的细胞核中心，同时对每个细胞核斑点进行检测，以排除一些干扰（>13$um^2$）。
图2展示了提出的深度残差聚合网络的网络结构。它遵循两个连续处理路径的着名范例：收缩路径（对输入进行下采样）和扩展路径（对收缩路径的输出进行上采样），例如U-Net（Ronneberger等，2015），SegNet （Badrinarayanan等，2017），FCN（Long等，2015）和Hypercolumns（Hariharan等，2015），有几个主要和次要的修改。
<img src="/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g0002.jpg" style="width:80%" />
#### 收缩路径
收缩路径可以看作是一个特征提取的步骤---识别目标的近似位置并编码他们的局部特征。为了这个目标我们使用了预激活的ResNet50,原始的ResNet的结构是：
卷积--批归一化---整流线性单元作为残差单元，预激活的结构是：批归一化--整流线性单元--卷积，同时通过快捷路径的方式，促进输入的直接传播。对预激活ResNet50做了一些修改：第一个7x7的无填充卷积是步幅为1，第一个7x7卷积后的最大聚集（池化）已被删除。

#### 膨胀路径
膨胀路径包含四个处理层，第一层接受收缩路径最后一层的输出，然后进行转换卷积和上采样。然后第二三四层接收两个输入，一个是上一层的输出，另一个是从收缩路径中来的。把两个输入加在一起，然后进行解码和进行大小调整。
大小调整单元是采用简单的最近邻插入法，就可以把大小翻倍，计算比较容易。
而且不采用连接的方式，采用附加运算符可以减少内存的使用，而不会降低网络的学习能力。
解码器是主要的处理单元，在解释来自不同抽象级别的信息并在膨胀路径中生成更精细的分割图方面起着关键作用。他通过采取多经架构，执行一系列卷积运算，其中输入和输出通道，被分成许多不相交的组（或路径），并且每个解码器分别执行卷积，所有的卷积操作不使用填充，步幅为1.由于没有填充的卷积分割图的大小越来越小。如表一所示，在我们的网络中使用了三个解码器，他们具有相同的结构，但是具有不同的通道和路径。
![表一](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-t0001.png)

#### 多尺度聚合
由于组织样本的大小不同，为了更好的提取特征和提高分割性能，采用多尺度的方法，样本大小被调整为0.5，1.0，2.0 倍。
调整好的图像分别进入DRANs 进行训练和准备，一共有3个DRAN，分别处理0.5,1.0，2.0倍的图像。DRAN的最后的softmax层被移除了。
解码器1通过另一个解码器4聚合，生成最终原始比例的分割图。要注意到解码器4使用的是填充卷积。具体细节如图3，这个网络叫做MDRAN。
![图三](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g0003.jpg)
Multiscale Deep Residual Network (MDRAN) architecture. MDRAN composes of 3 DRANs at 3 scales (x0.5, x1.0, x2.0) and a decoder (in dash rectangle), aggregating 3 scales together and generating a segmentation map at x1.0 scale. In the decoder, the convolution block [128, 5 × 5, 256] denotes [128 input channels, 5 × 5 kernel, 256 output channels].

### 两步的自动化方法对整个载片图像进行分类
之前的人提出来的方法都有一些缺点。我们想出来了一个非小细胞肺癌分类的新方法，这个方法主要集中在图片中决定癌症类型的诊断区域。
在2.2.1章节，我们描述了基于补丁的深度学习框架
2.2.2和2.2.3，我们描述了把整个载片图像分类为LUAD或者LUSC的随机森林回归模型。可以在图4中看整个分类的概述。
![图四](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g0004.jpg)
Overview of the NSCLC classification framework. (A) Workflow for training the neural network to classify input patches as either non-diagnostic (ND), lung adenocarcinoma (LUAD), or lung squamous cell carcinoma (LUSC). (B) Workflow for processing the WSIs within the test set to obtain probability maps for each class. (C) Workflow for the random forest regression model. Features are extracted from LUAD and LUSC probability maps and then fed as input into the random forest model. SN stands for stain normalization by method of Reinhard et al. (2001).

#### 网络架构
受到ResNet在图像识别任务中的成功的启发，我们实现了一个残差块为中心的深度神经网络来为NSCLC补丁分类。这个网络架构是ResNet50的一个变体。我们使用3x3的核代替7x7的核在第一次卷积的时候。在这个领域使用3x3的核是重要的，为了找到小特征，需要一个小的接受域，这是在组织图片中很常见的操作。使用3x3，可以使得参数数目变小，可以使得网络更广泛，降低过拟合的可能性。为了降低参数的数目，我们更改了ResNet50，减少了整个网络的残差块，使用32层代替50层。由于图像之间的高度可变性，因此在训练和验证集之间，考虑防止过度拟合是至关重要的。图5是网路架构的概述。图五是网络架构的概述。
![图四](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g0005.jpg)
训练完成后，我们选择对应的最好的平均验证准确度的阶段。然后处理每个测试WSI的补丁，得到三个概率图。
#### 统计和形态特征的提取
为了将每个WSI分类为肺腺癌或肺鳞状细胞癌，我们从LUAD和LUSC概率图中提取了特征。我们研究了两个后置处理技术：
最大投票和随机森林回归模型，最大投票模型只需要获取到LUAD和LUSC的正向的补丁数量就可以进行分类了。对于随机森林回归模型，我们从LUAD和LUSC训练概率图中都提取出了50个统计和形态特征。然后基于类的可分性选择了前25个特征。我们通过处理每个训练晚期的WSI来获取训练概率图。这个保证了网络已经对训练集过拟合，而且对LUAD和LUSC的诊断区域有一个好的分割。换句话说，使用这种方法允许我们从非穷举转换为穷举标记概率图。 一旦用这些特征训练模型，然后将它们作为特征输入到随机森林回归模型中。 提取的统计特征包括：概率图的均值，中值和方差。我们还计算了LUAD和LUSC概率图之间的比率。提取的形态特征包括在不同阈值处的前5个连接的组成部分的大小。
#### 随机森林回归模型
是一个把一些分类器结合在一起，来提高结果准确度的集成方法。一个例子就是随机森林，把多个决策树放在一起，来产出一个更好的分类结果。决策树根据某个参数连续分割输入数据，直到满足标准。 具体而言，随机森林回归模型让决策树拟合不同数据的子样本，然后计算所有决策树的平均输出。 我们采用了10个bagged 树来优化我们的随机森林模型，为每个决策分割随机选择三分之一的变量并将最小叶片大小设置为5.我们最终选择了一个阈值来转换随机森林回归模型的输出 转换为二进制值，表示WSI是LUAD还是LUSC。

## 结果
### 分割效果评估方法
我们组织了MICCAI 2017 数字病理学挑战，精挑细选了数据集。这些数据集都是从相关病人的组织图像。。。。
计算分割结果的得分，使用了DICE 系数和一个DICE系数的变体，叫做“Ensemble Dice”
DICE系数衡量了实际上和算法输出之间的重叠部分，但是没有考虑到分离和融合的情况，“分离”是本是一个核，算法输出是多个核，“融合”是本是多个核，算法输出一个核。使用DICE系数，一个算法把两个接触或者重叠的核看作一个和正确的分割为两个，具有相同的DICE。DICE-2可以解决“分割”的问题。伪代码如下：
![伪代码1](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-i0001.jpg)
Q and P are the sets of segmented objects (nuclei). The two DICE coefficients were computed for each image tile in the test dataset. The score for the image tile was calculated as the average of the two dice coefficients. The score for the entire test dataset was computed as the average of the scores of all the image tiles.

整个载片组织图像的分类，在病理学家的帮助下，获取到了32个案例，16个LUAD和16个LUSC。测试数据同样有32个，16个LUAD和16个LUSC。

### 分割细胞核的深度学习方法实验性评估
从原始的32个图像区块中，提取尺寸为200x200的多个区块，生成三种训练数据。通过滑动步长为54的窗口和剪切操作，生成了4732个图像块作为细胞核团（NBL）数据集。把细胞核放在图像块的中间，生成细胞核边界数据集，产生了2785个块。小核（SN）数据集是NBL的重复数据集，其仅包含具有≤50％核像素的核斑块。 SN数据集仅用于训练DRAN用于核斑点检测。
![表2](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-t0002.png)
在训练的时候，可以通过以下方法进行数据扩充：
1.考虑到补丁的宽度和高度，在[-0.05,0.05]范围内的随机垂直和水平位移
2.在[-45°，45°]度范围内随机旋转
3.随机垂直和水平翻转，概率为0.5
4.随机剪切，强度范围为[-0.4π，0.4π]
5.随机调整大小，比率范围为[0.6,2.0]
在数据增强之后，在图片进入网络之前，先把102x102中心区域提取出来（图片6）。对每个补丁进行三次增强。
![图6](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g0006.jpg)
Image patch generation. To avoid zero-padding in augmentation, a patch of size 200 × 200 is first provided. Subsequently, the center region of 102 × 102 is cropped and fed into the network as input. For an input of size 102 × 102, the network provides a segmentation map of size 54 × 54.
使用上面生成的训练数据，通过Adam优化器训练DRAN，其具有默认参数值（β1= 0.9，β2= 0.999，ε= 1e-8）在整个训练过程中保持32的小批量。L2正则损失也用来提高网络的广泛性。因子设为1.0e-5 K.He 初始化膨胀路径中的卷积层。训练过程分为两步，第一步（35个时代）预先激活的ResNet50的与训练权重加载到收缩路径，并保持不变（不更新权重）。在这个步骤中，只有在膨胀路径是可以训练的。学习率初始设定为1.0e-4，然后在0,1,15,25和35时刻变为5.0e-5,1.0e-5,7.5e-6和5.0e-6。在这个步骤中，使用NBL数据集训练细胞核斑点检测，NBD数据集进行边界检测。在第二步（40个时代），收缩路径解冻，整个网络都是可以进行训练的了。对于核斑点检测，NBL和SN数据集都用于进一步细化网络。 仅NBD数据集用于核边界检测。 此外，对损失函数施加不同的惩罚以减轻NBD数据集中的重偏差;每个边界像素的权重是5.0.把边界像素分类为背景像素的权重是6.0，背景像素的权重是1.0，对背景像素错误分类的惩罚是4.0。
在另一方面，和DRAN一样的训练数据集的多尺度模型训练的过程细节如下：MDRAN的每个DRAN分支加载有从上述过程获得的预训练DRAN权重并保持冻结。 然后，网络继续训练解码器4（10个时期），学习速率为1.0e-4。之后，在DRANs的膨胀路径解冻，微调35个时代，学习率分别在第1纪元，第15纪元和第30纪元设定为1.0e-4,1.0e-5和1.0e-6。
总体而言，利用NBL + SN数据集，MDRANBL被训练用于核斑点检测。 DRANBD接受了针对核边界检测的NBD数据集的训练。 结合两种模型（MDRANBL + DRANBD），进行细胞核分割。
表3展示了分割的结果：上面是没有使用多尺度，下面使用了多尺度。
![表3](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-t0003.png)
在测试集的比较中，多尺度聚合本质上提高了分割性能，特别在以20x放大率扫描的三个测试图片上。
对于以40倍放大率扫描的其他测试图像，多尺度聚合通常略微优于单一尺度方法。 这表明多尺度聚集特别有助于改善（相对）较小核的分割。 图8显示了多尺度聚合和单尺度方法的分割结果; 单尺度方法错过了几个小核，然而，这些核被多尺度聚集所识别。
图7：
![图7](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g0007.jpg)
Head-to-head comparison between MDRANBL and DRANBL on the test set. Test images are ordered by the ascending order of MDRAN DICE_1. The shaded area indicates that the images were scanned at 20x magnification.
图8：
![图8](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g0008.jpg)
Examples of nuclei segmentation via the multiscale aggregation (MDRANBL+DRANBD) and single scale (DRANBL+DRANBD) approach. The images from top to bottom are the 1st, 11th, 20th, and 25th image tile in the test set.
值得注意的是，在几个测试图像中观察到DICE_1和DICE_2之间存在巨大差异（图7）。 经过仔细检查，我们发现这些主要是由于染色变化和不稳定以及核密集重叠。 如图9所示，所识别的核边界通常是碎裂的和不完美的，导致重叠核的不准确分割。 这表明先进而精密的接触核分离方法可能具有提高分割性能的巨大潜力。
![图9](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g0009.jpg)
Examples of correct and incorrect nuclei segmentation. Our method (Bottom) is able to distinguish the boundary of the non-highly overlapping nuclei fairly well but (Top) fails on the highly overlapping nuclei with disproportionate stains.

### 整个载玻片图像的分类

我们使用了总共64个染色后的NSCLC 载片图像。我们把它们分为32个训练和32个测试图像。
我们平均分了NSCLC图像，得到32个LUAD载片和32个LUSC载片。划分出24个WSI作为训练，8个作为验证。
我们从病理学家确认后的非穷尽的标记区域提取了3类数据，他们由256x256，放大20倍的补丁组成。
三类数据集包含，LUAD，LUSC，和ND（non-diagnostic）。总体而言，我们的网络针对65788个训练图像进行优化。
由于是从不同的中心获取的图像，所有图像之间存在很高的染色变化。为了解决这个问题，我们应用了Reinhard等人的方法，通过把图像映射到预先定义图像的统计数据来对所有图像进行归一化。在训练过程中，我们执行随机裁剪，翻转，和旋转数据增强。以使得对这些变换不变。在对所有的输入补丁进行随机裁剪之后，我们留下224X224大小的补丁。
残余网络背后的直觉是优化残差映射比优化原始未引用映射更容易。 残差块是ResNet的核心组件，由前馈跳跃连接组成，它执行身份映射，无需添加任何额外参数。 这些连接在整个模型中传播梯度，这反过来使得网络能够被更深地训练，通常实现更高的准确性。

表4总结了我们为将输入补丁分类为LUAD，LUSC和ND而进行的实验。 我们选择训练指定的网络，因为它们在最近的图像识别任务中具有最先进的性能（Deng et al。，2009）。 在训练期间，所有网络都快速过拟合训练数据。 这是因为两个原因：（i）所使用的网络架构已针对具有数百万图像和数千个类的大规模计算机视觉任务进行了优化; （ii） 由于我们的数据集的大小，我们的训练集规模相当有限。（iii）在给定足够数量的模型参数的情况下难以避免过度拟合。 因此，我们修改网络架构以通过减少层数来解决过度拟合的问题。 我们通过减少剩余单元的数量来修改ResNet的原始实现，这样我们在模型中总共只有32层
![表4](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-t0004.png)
由于具有出色的补丁级性能，我们选择使用ResNet32处理测试集中的图像。 图10显示了四个测试WSI及其重叠概率图。 绿色区域显示分类为LUSC的区域，蓝色/紫色区域显示分类为LUAD的区域，黄色/橙色区域显示分类为ND的区域。
![图10](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-g00010.jpg)
Test WSIs with overlaid probability maps. Blue/purple indicates a region classified as diagnostic LUAD, green indicates a region classified as diagnostic LUSC, yellow/orange refers to a region classified as non-diagnostic.
表5显示了由挑战组织者处理的NSCLC WSI分类的总体准确度。 我们观察到使用具有来自标记的WSI的统计和形态特征的随机森林回归模型提高了分类准确性。 当LUAD或LUSC是标记的WSI中的主导类时，最大投票就足够了，但是当没有明显的优势类时，随机森林回归模型会提高性能。 这是因为用作随机森林模型的输入的特征比仅使用投票方案更具信息性，因此可以更好地区分每种癌症类型。
![表4](/images/论文阅读笔记/Methods_for_Segmentation_and_Classification_of_Digital_Microscopy_Tissue_Images/fbioe-07-00053-t0005.png)
上面提出的方法，达到了0.81的好成绩，在数据集有限的情况下，可以发现把NSCLC分为诊断区域和非诊断区域是非常重要的。因为这些训练数据不包含正常的案例，所以对这个的特殊实现是重要的。如果不把非诊断区域纳入考虑范围，算法就会被强制对非信息图片块进行预测。更进一步，这证明了分类区域的形态可以为整个载片图像的分类带来帮助，并且优于最大投票方法。对上下文信息的考虑可以为计算病理学中的分类任务提供额外的帮助（Agarwalla等，2017; Bejnordi等，2017）。例如，在分类NSCLC病例时，LUAD病例中的生长模式以及肿瘤如何随基质生长是非常重要的。在分辨率为20×的224×224补丁中，这些模式通常非常难以可视化。
在未来的工作中，开发出包含更多上下文信息的网络可能对提高补丁级别的分类有帮助，因此，也可以提高整个分类的准确度。为了开发这项工作，我们还旨在使用更大的数据集，以便我们的补丁级分类器能够为随后的NSCLC分类提取更具代表性的特征。






