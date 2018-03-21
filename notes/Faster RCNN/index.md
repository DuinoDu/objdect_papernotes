---
title: Faster R-CNN
date: "2018-03-15T07:52:00.000Z"
publish_time: "2015.06"
compare_method: [1]
---

### 1. 问题

如何提升region proposal的计算效率和质量？

### 2. 做法

用神经网络来做region proposal提升质量，同时通过共享权重，极大减少proposal的计算时间。



[^1]: Scalable, high-quality object detection（MultiBox）