---
title: Mask RCNN
date: "2018-03-13T00:00:00.000Z"
---


### 1. 问题
如何把FasterRCNN框架用于分割任务?

### 2. 做法
目标检测和分割的区别在于，对目标的表达不同：目标检测认为目标只要用一个矩形框框住目标就行，而分割用目标的精细轮廓来表达目标。从结果来看，似乎目标检测相比于分割，更粗略。

另一方面，之前检测和分割，是两个独立的问题，尤其是在deeplearning之前，目标检测的思路是滑动窗口+特征提取+分类器，而分割的思路，比如超像素、分水岭、种子分割、主动轮廓等，研究方法完全不同，解决问题的思路也不一样。

而随着deepleaning在图像分类任务上取得的重大突破[^1], 目标检测领域出现了一些开拓性的工作[^2,3,4,5], 分割领域也出现了一篇开篇工作[^6]。


[^1]:Imagenet classification with deep convolutional neural networks 
[^2]:Deep neural networks for object detection
[^3]:Rich feature hierarchies for accurate object detection and semantic segmentation
[^4]:Scalable Object Detection using Deep Neural Networks
[^5]:OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks
[^6]:Fully Convolutional Networks for Semantic Segmentation
