import numpy as np
import matplotlib.pyplot as plt
from homework4_sbuasai2 import SVM4342

def normalToSlope (normal_vector):
    x,y = normal_vector
    return -x/y

X = np.load("hw4_X.npy")
y = np.load("hw4_y.npy")
n = X.shape[1]//2

x = np.arange(-8, +8, 0.01)
plt.scatter(X[0,0:n], X[1,0:n])
plt.scatter(X[0,n:], X[1,n:])

# H1
m1 = 2/3
b1 = 1
w1 = np.array([0,1])*m1
slope1 = normalToSlope(w1)
y1 = 0*x-3/2
y_up1 = 0*x+0
y_down1 = 0*x-3

plt.plot(x, y1, 'k-')
plt.plot(x, y_up1, 'k--')
plt.plot(x, y_down1, 'k:')

# H2
m2 = 2.85714
b2 = 2.42857
w2 = np.array([-0.3,1])*m2
slope2 = normalToSlope(w2)
y2 = 0.3*x-0.85
y_up2 = 0.3*x-0.5
y_down2 = 0.3*x-1.2

plt.plot(x, y2, 'b-')
plt.plot(x, y_up2, 'b--')
plt.plot(x, y_down2, 'b:')

# H3
Wtilde = SVM4342()
Wtilde.fit(X.T, y)
b3 = Wtilde.b
w3 = Wtilde.w
slope3 = normalToSlope(w3)
y3 = -0.285714*x-2.35714
y_up3 = -0.285714*x-(9.42857e-8)
y_down3 = -0.285714*x-4.71429

plt.plot(x, y3, 'r-')
plt.plot(x, y_up3, 'r--')
plt.plot(x, y_down3, 'r:')
plt.show()
