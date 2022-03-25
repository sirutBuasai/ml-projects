#!usr/bin/env python3
import numpy as np

a = np.array([1,2,3])
b = np.atleast_2d(a)
x = np.array([np.ones(a.shape)])
for i in range(1,4):
    x = np.vstack((x,a**i))
print(x)
print(a)
print(b)
