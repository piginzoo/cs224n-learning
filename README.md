这个是CS224n课的作业！
是跟着https://github.com/learning511/cs224n-learning-camp 每周学习进行的！

# 作业1 第一周
assiment1/q1_xxx 和 q2_xxx
没做完，主要是不理解q2_gradcheck.py到底要干啥？
是为了做递归地梯度下降么？
另外，cs224中并没有讲深度网络的BP，所以对softmax,sigmod函数的导数啥的概念都没有提，但是这个程序里都要用，要手工推一遍，自己。参考：https://blog.csdn.net/Jaster_wisdom/article/details/78379697

另外，可以熟悉了一下多维数数组中的broadcasting，http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html ，还是很有用的。

别的没啥了，这个作业基本上达到了目的。

# 作业2 第二周
assiment1/q3_xxx

## 使用了哪些材料：
首先要说，这个[作业辅助笔记](https://github.com/piginzoo/cs224n-learning/blob/master/solution/assignment1%20(1.1%2C1.2%2C1.3%2C1.4)/assignment1_soln.pdf),其次是网上流传的那篇著名的《[word2vec数学原理](https://github.com/piginzoo/cs224n-learning/blob/master/assignment1/word2vec_maths.pdf)》，再有就是视频课程了。

这个是作业的帮助，我其实没有写代码，真心写不出来，完全没思路呢，课还听得一知半解呢。
后来从网上发现老师发的课件里居然有代码，很高兴，认真阅读了一下。
挨行写了注释，终于搞明白了，把思路下载这里，也算是对学习的一个帮助把。

## 启动是q3_run.py
```
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(
        skipgram, #skipgram算法的函数指针
        tokens, #词表
        vec, #每行的词向量？还是每一列？干嘛？？？
        dataset, 
        C,
        negSamplingCostAndGradient),#负采样的时候的梯度？
    wordVectors, #词向量矩阵，别忘是2个词表矩阵合成的（2D * V）

```
这种lambda写法很诡异，为此我专门写了一个 [test_lambda.py](https://github.com/piginzoo/cs224n-learning/blob/master/assignment1/test/test_lambda.py),来捋清楚这个。
其实核心就是为了让核心参数wordVectors晚点绑定到函数里而已。

## [q3_sgd.py](https://github.com/piginzoo/cs224n-learning/blob/master/assignment1/q3_sgd.py)

然后sgd()方法没干啥，就是触发了word2vec_sgd_wrapper而已，然后word2vec_sgd_wrapper里面调用了skigram，skigram里面调用了negSamplingCostAndGradient。大致这么一个调用链。

```sgd()->word2vec_sgd_wrapper()->skimgram()->negSamplingCostAndGradient()```

## [q3_word2vec.py](https://github.com/piginzoo/cs224n-learning/blob/master/assignment1/q3_word2vec.py#1)

先看看word2vec_sgd_wrapper：
- 没有遍历整个语料，而是随机去找一个词，然后用它来形成一个窗口来做预测，这个玩法很诡异
- 然后他调用word2vecModel(),其实就是skimgram()

然后看看skipgram:
- 对每个上下文词进行处理，调用word2vecCostAndGradient，其实就是negSamplingCostAndGradient
- 针对的是一个词，一个中心词，一个上下文词

然后看看negSamplingCostAndGradient:
- 这个类很很核心，其中的梯度公式很重要，要[参考](https://github.com/piginzoo/cs224n-learning/blob/master/solution/assignment1%20(1.1%2C1.2%2C1.3%2C1.4)/assignment1_soln.pdf) 3.word2vec (c)
- 3(c)中关于v_c, u_o, u_w的梯度公式很重要，这些就是词向量矩阵的U,V的待定系数呀


