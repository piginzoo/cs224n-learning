#encoding=utf-8
import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    D-dimensional vector (treat the vector as a single row) and
    for N x D matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    print "输入=",x
    x = x - np.max(x)
    print "输入(-max)=",x

    if len(x.shape) > 1:
        print "多维：",x.shape
        # Matrix
        ### YOUR CODE HERE
        #no need to re-create ndarray,just use x in-place
        # result = numpy.ndarray(x.shape)
        print("e=",np.exp(x),",shape=",np.exp(x).shape)
        _sum = np.sum( np.exp(x),axis=1)
        print("sum=",_sum,",shape=",_sum.shape)
        x = np.exp(x).T / np.sum( np.exp(x),axis=1).T
        x = x.T
        
        ### END YOUR CODE
    else:
        print "一维：",x.shape
        # Vector
        ### YOUR CODE HERE
        #关于sum：https://blog.csdn.net/ikerpeng/article/details/17026011
        x = np.exp(x) / np.sum( np.exp(x))
        ### END YOUR CODE
    print "softmax结果=",x
    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."

    test1 = softmax(np.array([1,2]))    
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[11,12],[3,4]]))
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
