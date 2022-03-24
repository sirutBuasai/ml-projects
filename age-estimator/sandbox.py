#!usr/bin/env python3
import numpy as np

a = np.array([[[1,1],[1,1]],[[2,2],[2,2]],[[3,3],[3,3]],[[4,4],[4,4]]])
b = np.reshape(a, (a.shape[1]*a.shape[2], a.shape[0]))
one = np.ones(b.shape[1])
c = np.vstack((b, one))
print(a)
print(b)
print(one)
print(c)
