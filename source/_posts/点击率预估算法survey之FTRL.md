---
title: 点击率预估算法survey之FTRL
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

这里使用python以及golang分别实现ftrl算法。使用的数据下载自kaggle的train、test数据。大部分逻辑是参考以github相关开源的实现。记录下来供参考。

数据的格式为csv的格式，我下载的数据内容为这里[avazu点击数据](https://www.kaggle.com/c/avazu-ctr-prediction/data)，压缩后数据大小为1G多，包含有40428968行。这里给出具体的实现内容并分

析。

```pyt

from csv import DictReader 
from math import exp, log, sqrt
from datetime import datetime

import sys

D = 2 ** 20

train="../py-ftrl-kaggle/small.csv"
train = sys.argv[1]

### since the date in data is 21-30
### then if date = 30, then take it as the validate data to calc the logloss
holdafter = 9
holdout = 100

class FTRL:
    ## l1, l2
    def __init__(self, l1, l2, alpha, beta, epoch, interaction):
        self.n = [0.] * D
        self.w = {}
        self.z = [0.] * D

        self.l1 = l1
        self.l2 = l2
        self.epoch = epoch
        self.interaction = interaction

        self.alpha = alpha
        self.beta = beta


    def data(self, file, D):
        for t, row in enumerate(DictReader(open(file))):
            ID = row["id"]

            ## delete row["id"] for memory saving
            del row["id"]

            y = 0.
            if "click" in row and row["click"] == "1":
                y = 1.

            ## delete row["click"] for memory saving
            del row["click"]

            ## data oriented, the train data use day 21-30 for train
            date = int(row["hour"][4:6])
            date -= 20

            ## write useful hour data
            row["hour"] = row["hour"][6:]


            ## for each value ,generate its index
            x = []
            for key in row:
                value = row[key]

                index = abs(hash(key + "_" + value)) % D
                x.append(index)
                
            
            yield t, date, ID, x, y



    ## for instance x, produce cross feature if need interaction
    def fullindex(self, x):
        
        ## for bais index
        yield 0

        for index in x:
            yield index

        ## if support interaction
        if self.interaction:
            l = len(x)
            x = sorted(x)
            for i in xrange(l):
                for j in xrange(i+1, l):
                    yield abs(hash(str(x[i]) + "_" + str(x[j]))) % D


    ## handle one instance x
    def predict(self, x):
        wtx = 0.
        z = self.z
        n = self.n

        a = self.alpha
        b = self.beta
        l1 = self.l1
        l2 = self.l2
        
        w = {}

        wtx = 0
        ## maybe has cross feature
        for i in self.fullindex(x):
            sign = -1. if z[i] < 0 else 1.

            if sign * z[i] <= l1:
                w[i] = 0.
            else:
                w[i] = (sign * l1 - z[i])/((b + sqrt(n[i]))/a + l2)
        
            wtx += w[i]

        ## get updated w
        self.w = w

        return 1. / (1. + exp(-max(min(wtx, 35.), -35.)))

    ## for instance x, calculate the p 
    ## using g to update n and z
    def update(self, x, p, y):
        a = self.alpha
        b = self.beta
        n = self.n
        z = self.z

        w = self.w

        ## here actually g = (p - y) * x, and x here is 1.
        g = p - y

        ## if has interaction, update the n and z for cross features
        for i in self.fullindex(x):
            sigma = (sqrt(n[i] + g*g) - sqrt(n[i]))/a
            z[i] = z[i] + g - sigma * w[i]
            n[i] = n[i] + g * g


    ## hope the p in the gap [10e-15, 1-10e-15]
    def logloss(self, p, y):
        p = max(min(p, 1. - 10e-15), 10e-15)
        return -log(p) if y == 1 else -log(1. - p)

    def run(self):
        start =  datetime.now()
        learner = FTRL(l1=1., l2=1., alpha=0.1, beta=1., epoch=1, interaction=False)

        logcount = 0
        totalloss = 0.
        for t, date, ID, x, y in self.data(train, D):

            p = self.predict(x)
            #print "p:", p

            ## date is 1-10
            if date > holdafter:
                print "date:",date,"holdafter:",holdafter
                logcount += 1
                loss = self.logloss(p, y)
                totalloss += loss

            else:
                self.update(x, p, y)

        if logcount != 0:
            print "avg logloss:", totalloss / logcount, \
                    "logcount:", logcount,\
                    "time elapsed:", datetime.now() - start
def main():

    ftrl = FTRL(l1=1., l2=1., alpha=0.1, beta=1., epoch=1, interaction=False)
    ftrl.run()

## run
main()
```

我们跑了10w个数据，结果如下：

```bash
date: 10 holdafter: 9
date: 10 holdafter: 9
avg logloss: 0.160171832485 logcount: 2 time elapsed: 0:00:07.045008
```

这里需要说明，date这一列的数据形式为14102100，即YY-MM-DD-HH，其中Hour设置为0，train数据中日期的分布是1021-1030。这里我们取出date后再减去20，得到的日期范围为1-10。其中10我们不参与训练，而是用在计算logloss上，作为model的validate。



代码初始花，我们设置了$n,z$两个变量用来存储训练中的迭代变量。

$x$对于一行instance，读取文件后，基于hash() %D，作为这列特征值的id，并以这个id作为索引在$n,z$中读写。

其中提供的fullindex(self, x)是基于原始$x$，构建cross特征的id。代码中设置任何两个变量之间进行cross。当然这里没有做太多的特征工程，只是简单的cross了所有特征。

如果interaction=False，那么训练的还是原始数据instance。

predict以及update严格按照google的paper中给出的算法

### 结语

相对于之前使用的LBFGS，或OWLQN，最大的不同是以前的的算法会严格提出stopping criterion。但是FTRL并没有这么做。我理解，首先作者已经给出了证明，这个算法是必然收敛的；其次这里给出的是简单的遍历所有instance进行train，实际的工程使用中，可能有其他的策略变化，这里待补充实际的工业实践。

