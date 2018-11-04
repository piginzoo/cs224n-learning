#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    denom = np.apply_along_axis(lambda x: np.sqrt(x.T.dot(x)), 1, x)
    x /= denom[:, None]
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print x
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    #要被预测是词向量，也就是的中心词
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    #目标词汇，是一个索引号，啥叫target，就是要对应的上下文词             
    target -- integer, the index of the target word
    #就是W',右面的那个词向量组成的矩阵，
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    #整个得到的损失函数，是一个交叉熵的值
    cost -- cross entropy cost for the softmax word prediction
    #对预测的那个词，也就是中心词的，是一个未知向量，也就是参数，他的梯度
    gradPred -- the gradient with respect to the predicted word
           vector
    #别的词，也就是context词的梯度
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    ## Gradient for $\hat{\bm{v}}$:

    #  Calculate the predictions:
    vhat = predicted #输入，就是要预测的词，一个词表V长度的一个向量，很长
    #z是啥？
    #outputVectors是一个矩阵，vhat是
    z = np.dot(outputVectors, vhat)#outputVector就是W'，就是输入词向量要乘以的那个矩阵
    preds = softmax(z)#z是得到一个W' * v_c，是一个词表大小的向量，但未归一化，需要归一化
                      #然后用softmax将其归一化，得到一个归一化的预测向量，
                      #也就是预测是某个词的概率        

    #  Calculate the cost:
    cost = -np.log(preds[target])#交叉熵

    #  Gradients
    #参考：https://github.com/piginzoo/cs224n-learning/blob/master/solution/assignment1%20(1.1%2C1.2%2C1.3%2C1.4)/assignment1_soln.pdf
    z = preds.copy()
    z[target] -= 1.0#计算梯度，是对z是

    #outer计算是算一个矩阵，
    grad = np.outer(z, vhat) #这步实现 vhat * (z - 1)
    gradPred = np.dot(outputVectors.T, z) 
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K #
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

'''
在skimgram中会真正调用他：
word2vecCostAndGradient(
    vhat, #中心词的那个词向量
    u_idx,#其他的上下文词的在词表中的位置index，后面会用来找到他对应的词向量 ，是一个整数值
    outputVectors, #输出矩阵
    dataset)#数据集
'''
def negSamplingCostAndGradient(
    predicted, #被预测向量,就是v_c: http://www.hankcs.com/nlp/word-vector-representations-word2vec.html
    target, #上下文词的索引，起名叫target我觉得是指对应要预测他们,就是 u_i
    outputVectors, #输出矩阵，也就是老师讲的W'
    dataset,
    K=10):#K??我没看呢，现在理解是采样负样本的个数？，对，靠，下面写着呢： K is the sample size.
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]#为何又加了个括弧了？？？[3]的感觉了？
    #这个肯定是之前的10个正样本+了10个负样本
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    cost = 0
    #就是 sigmod(中心词 * 上下文词),z得到一个概率
    z = sigmoid(np.dot(outputVectors[target], predicted))

    cost -= np.log(z) #这个好理解，就是交叉熵，别的都为0，就这个维度为1： -p*logq, p现在是1，q现在是z
    
    #这个梯度是对u_0（正确的分类的那个词）的梯度，也就是outputVectors里面参数的梯度
    #这个是经过3.c中的采样梯度公式，具体参考这个文档的 3(c)
    #https://github.com/piginzoo/cs224n-learning/blob/master/solution/assignment1%20(1.1%2C1.2%2C1.3%2C1.4)/assignment1_soln.pdf
    grad[target] += predicted * (z - 1.0) #predicted就是v_c center word，中心词
    #这个是对v_c(中心词)的梯度       
    gradPred += outputVectors[target] * (z - 1.0)
    #这个是算负样例的梯度，实际上是u_k（负样例）
    for k in xrange(K):
        samp = dataset.sampleTokenIdx()
        z = sigmoid(np.dot(outputVectors[samp], predicted))
        cost -= np.log(1.0 - z)
        grad[samp] += predicted * z
        gradPred += outputVectors[samp] * z
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    cword_idx = tokens[currentWord]
    vhat = inputVectors[cword_idx]#找到中心词对应的D维词向量

    for j in contextWords:#然后对每一个包围这个中心词的上下文词
        u_idx = tokens[j]
        c_cost, c_grad_in, c_grad_out = \
            word2vecCostAndGradient(
                vhat, #中心词的那个词向量
                u_idx,#其他的上下文词的在词表中的位置index，后面会用来找到他对应的词向量 
                outputVectors, #输出矩阵
                dataset)#数据集
        cost += c_cost
        gradIn[cword_idx] += c_grad_in
        gradOut += c_grad_out
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    predicted_indices = [tokens[word] for word in contextWords]
    predicted_vectors = inputVectors[predicted_indices]
    predicted = np.sum(predicted_vectors, axis=0)
    target = tokens[currentWord]
    cost, gradIn_predicted, gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    for i in predicted_indices:
        gradIn[i] += gradIn_predicted
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################
#C是窗口大小，默认是5，5-1-5
#word2vecModel：skimgram算法函数
#tokens：词表
#wordvectors
def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)#2D*V的词表矩阵全部初始化为0
    N = wordVectors.shape[0]#词表长度
    inputVectors = wordVectors[:N / 2, :]#[行:列]，前N行，是输入矩阵
    outputVectors = wordVectors[N / 2:, :]#后N行，是输出矩阵
    for i in xrange(batchsize):#一个batch 50次
        C1 = random.randint(1, C)
        #这句话很诡异，实际上如果去看utils/treebank.py:95的getRandomeContext函数
        '''
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)<-----全局随机数
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)
        context = sent[max(0, wordID - C):wordID]        
        '''
        #看到了把，实际上是在全局随机捕捉目标文字
        #我还奇怪呢？怎么也看不到一个遍历整个语料的方式来训练呢？原来是在这里呢
        #所以他们用mini-batch就是50个凑一波，然后做GD
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(#<------就是skimgram函数
            centerword, #中心词
            C1, #窗口大小
            context, #这个窗口中的词
            tokens, #整个语料
            inputVectors, #输入矩阵
            outputVectors,#输出矩阵
            dataset, #整个语料，不知道还要这个玩意干嘛？
            word2vecCostAndGradient)#虽然默认是softmax哈夫曼实现，实际上后来传进来了负采样实现，
                                    #也就是negSamplingCostAndGradient
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in xrange(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    print skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
               negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
