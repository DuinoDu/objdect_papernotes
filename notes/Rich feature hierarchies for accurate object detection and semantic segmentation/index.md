---
title: Rich feature hierarchies for accurate object detection and semantic segmentation
date: "2018-03-12T00:00:00.000Z"
publish_time: "2013.12"
compare_method: [6]
---

 #### 1. 问题

2012年ImageNet图像分类比赛上，CNN取得重大突破，如何把这种算法上的优势，用在PASCAL VOC比赛上？

#### 2. 做法

**分析问题** 要回答这个问题，需要解决两个问题：

1. 如何用深度网络定位目标？也即，图像分类和目标检测有什么区别？
2. 如何用有限的数据量训练神经网络？（ImageNet有数百万张图片，VOC只有5千张）


CNN处理目标检测问题，有两种大体思路：

- regression，有一篇文章[^5]就是这么做的，但是效果不好，在很早的时候，就被研究社区抛弃了。
- classification，基于滑动窗口，几乎所有的方法，都是采取这种思路。
  - 直接使用滑动窗口（Overfeat）
  - 使用处理得到的region（RCNN)

这个区分，就直接导致了目前的目标检测的两大类别：一阶段、二阶段。而且在2013年就已经很明确了，Overfeat速度快，但精度不高，RCNN速度慢，但精度高，Overfeat比RCNN快9倍，精度低7个百分点。（RCNN论文的4.6节有两个方法的对比分析）

随着selective search和RCNN两个方法的出现，detection proposals一时称为当时的研究热点，出现了一系列计算目标候选区域的方法以及综述性的文章[^7]

- Selective Search
- Edge Boxes
- BING, BING++
- Objectness
- RPN
- 其他

直到RPN[^8]的提出，这方面的热度才慢慢减弱。

[^1]: Discriminatively trained deformable part models
[^2]: Selective search for object recognition
[^3]: Regionlets for generic object detection
[^4]: Bottom-up segmentation for top-down detection
[^5]:  Deep neural networks for object detection
[^6]: OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks
[^7]: How good are detection proposals, really?
[^8]: Faster R-CNN: Towards Real-Time Object

