import numpy as np
# (2, 3) (3,)
# [[1 1 1]
#  [2 2 2]]
a = np.array([[2,3,5],[4,6,10]])
b = np.array([2,3,5])
print a.shape, b.shape
print a/b

a = np.array([[2,4,6],[3,6,9]]).T
b = np.array([2,3]).T
print a.shape, b.shape
print (a/b).T

