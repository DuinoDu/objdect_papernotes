title: Fast R-CNN
date: "2018-03-14T00:00:00.000Z"
publish_time: "2015.04"
compare_method: [1]

### 1. 问题

本文要解决的问题是，如何让RCNN快一些？

### 2. 做法

RCNN最耗时的地方体现在，对于每一个proposal，都需要前馈计算一遍卷积网络。为什么RCNN需要对每一个proposal计算一遍前馈网络？因为在RCNN中，CNN的作用仅仅是特征提取器，输入滑窗内的像素，输出计算出的深度特征，然后给到SVM分类以及位置回归器（用最小二乘回归位置是DPM中使用的，在RCNN论文的附录C中详细阐述了具体做法，并取名为bounding-box regression）。

直接的想法是，让网络先对整张图像计算深度特征图，然后在深度特征图上找每个proposal对应的特征。这么做存在一个问题：RCNN的输入是固定大小的（227x227），所以在RCNN中，对各种不同大小的proposal，需要做一些处理转换成224x224（RCNN论文的附录A给出了几种转换方法），然后再用卷积网络提取特征。由于现在的proposal是从特征图上来，所以需要在特征图上，进行一些转换处理，得到固定大小的特征图。

SPPnet[^1]就是一个在特征图上通过一些转换操作，得到固定大小特征的方法，但SPPnet存在如下问题：

- 特征需要写到硬盘上缓存。
- SPP层之前的卷积层无法参与训练。

也就是说，SPP层让前后两部分截断了，为什么呢？（SPPnet的论文认为，SPP是一种类似Bag-of-Word的操作，输入可变长特征，输出定长特征，同时SPP还能保留空间位置信息，但是SPPnet并没有提供误差后向传播的方法，是不可训练的，遗憾的是，SPPnet文中并没有指出这一点。如果用类似pytorch具有自动求导功能的代码实现SPPnet，是否就能end2end训练了？应该是这样，那FastRCNN中提到的SPPnet的缺点，只是当时存在的缺点，现在应该不存在了，也就是说，目前的架构中，依然可以使用SPPlayer）

本文最大的贡献，是第一个在region-based的目标检测方法中，提出的端到端训练的模型。像Overfeat、YOLO、SSD等一阶段方法，很容易实现端到端模型。而这之前，RCNN和SPPnet，训练过程有一下几步：

1. 用ImageNet预训练模型
2. 提取特征
3. 用带log-loss的网络微调卷积网络
4. 训练SVM
5. 训练regressor

而本文所有的工作都在一次完成。

![](model.png)

#### 2.1 RoI pooling or SPPlayer?

SPPlayer和RoI pooling相比，多了一些尺度，这样让proposal计算的特征，具有不同尺度的特征。







[^1]: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition

