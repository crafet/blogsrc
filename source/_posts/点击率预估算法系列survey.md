---
title: 点击率预估算法系列survey
date: 2017-09-25 10:44:58
tags: ["ctr预估", "训练算法"]
categories: tech
---

 

### 论文主要内容

参考的paper是<Ad Click Prediction: a View from the Trenches>。文章中给出了可以工程化的实践。

在LR算法训练中，在第t个instance上的logloss为

​	$logloss\_t(w\_t) = -y\_tlog p\_t - (1-y\_t)log(1-p\_t)$

其中$w_t$是训练当前这轮的权重，$y_t$是label，$p_t$是预估的结果，即sigmoid函数的计算结果。一般$y_t$是0或者1。因此这个公式可以简化为

​	$$ logloss\_t(w\_t)= \begin{cases} -log(1-p\_t), & \text {if $y\_t=0$} \\ -logp\_t, & \text{if $y\_t=1$} \end{cases} $$

从而推倒出的gradient为

$g = (p\_t-y\_t)x\_t$，而instance在预处理之后对应的$x_t$一般是libsvm的数据格式，即$featid:1$，所以$x_t$的值一般是1，因此这里可以等价为$g = p\_t-y\_t$。



在FTRL算法中，这个是$g$是迭代的主要变量。$n,z$均是基于这个$g$进行构建。

Online Gradient Descent(OGD)被证明是有效的，但是不不擅长产出sparse模型。而在具体的线上服务，计算pctr的时候，sparsity决定了服务性能；并且简单的对logloss增加$L_1$正则化参数并不能非常有效的产出0值权重；

像FOBOS类似的复杂算法是简单粗暴的砍掉接近0值的权重以产生sparsity。

RDA算法在sparsity和accuracy之间取的平衡，而google提出的FTRL算法比RDA有更好的效果(更sparse的基础上更有accuracy)。



当没有正则化参数时候，FTRL算法等价于OGD，但是FTRL中使用了称为lazy representation of model coefficient $w$，因此实现$L_1$正则上更有效率。

对所有coordinate设置相同的学习率并不是很理想的。因此文章提出来per-coordindate learning rate。

这里贴出完整的ftrl算法



![ftrl算法](/images/ftrl-algo.png)

一般的实现上严格按照这里的表达式进行计算即可。在具体的实践中有几个trick：

第一个即sigmoid函数结果有个上限。

> return 1.0 / (1.0 + math.Exp(-math.Max(math.Min(w, 35), -35)))

即累加值$\sum(w*x)$设置在$[-35, 35]$之间即可。

### 工程实现

这里使用python以及golang分别实现ftrl算法。使用的数据下载自kaggle的train、test数据。

1. python实现

```pyt
class FTRL
```

