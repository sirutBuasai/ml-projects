#!usr/bin/env python3
import numpy as np

a = np.array([[[1,1],[1,1]],[[2,2],[2,2]],[[3,3],[3,3]]])
b = np.reshape(a, (a.shape[0], a.shape[1]*a.shape[2]))
c = np.column_stack((b, np.ones(b.shape[0])))
d = c[:,:-1]
e = d.reshape((3,2,2))
print(a)
print(b)
print(c)
print(d)
print(e)