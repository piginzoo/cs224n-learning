#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from q3_word2vec import *
from q3_sgd import *

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens() #？应该是词表
nWords = len(tokens) #词表个数？

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10 #10维的词向量

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

startTime=time.time()
#一个 词表矩阵 + 词表矩阵，拼接到一起，是2*词表行数 * 词向量维度的矩阵
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) /
       dimVectors, np.zeros((nWords, dimVectors))),
    axis=0)
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(
        skipgram, #skipgram算法的函数指针
        tokens, #词表
        vec, #每行的词向量？还是每一列？干嘛？？？
        dataset, 
        C,
        negSamplingCostAndGradient),#负采样的时候的梯度？
    wordVectors, #词向量矩阵，别忘是2个词表矩阵合成的（2D * V）
    0.3, #step参数，没搞清楚，是步长么？
    40000, #4万次迭代
    None, True, PRINT_EVERY=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print "sanity check: cost at convergence should be around or below 10"
print "training took %d seconds" % (time.time() - startTime)

# concatenate the input and output word vectors
wordVectors = np.concatenate(
    (wordVectors[:nWords,:], wordVectors[nWords:,:]),
    axis=0)
# wordVectors = wordVectors[:nWords,:] + wordVectors[nWords:,:]

visualizeWords = [
    "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"]

visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in xrange(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('q3_word_vectors.png')
