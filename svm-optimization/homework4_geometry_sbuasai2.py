import numpy as np
import matplotlib.pyplot as plt
from homework4_sbuasai2 import SVM4342

## Convert normal vector to slope of y = mx + b
def normalToSlope (normal_vector):
    x,y = normal_vector
    return -x/y

## Given an arbitrary normal vector and some data points,
#  find the normal weight vector w and bias b
def findWeight (normal, coord_up, coord_down):
    n_1, n_2 = normal
    x_1, y_1 = coord_up
    x_2, y_2 = coord_down
    # Equation for upperbound: x_1*(m*n_1) + y_1*(m*n_2) + b - 1 = 0
    # Equation for lowerbound: x_2*(m*n_1) + y_2*(m*n_2) + b + 1 = 0
    # Solve for m and b given the two equations
    m = 2/(n_1*(x_1-x_2) + n_2*(y_1-y_2))
    b = 1 - x_1*m*n_1 - y_1*m*n_2
    # normal weight vector w is m multiple of normal vector (w = m*n)
    w = np.array([n_1,n_2])*m
    return w, b

## Given x, w and b, find the equation y = mx + b
def findEquation (x, w, b):
    w_1, w_2 = w
    # Equation: w_1*x + w_2*y + b = 0,1,-1
    # Solve for y given the equation above
    y = (-w_1*x - b) / w_2
    y_up = (-w_1*x - b + 1) / w_2
    y_down = (-w_1*x - b - 1) / w_2
    return y, y_up, y_down

X = np.load("hw4_X.npy")
y = np.load("hw4_y.npy")
n = X.shape[1]//2

x = np.arange(-8, +8, 0.01)
plt.scatter(X[0,0:n], X[1,0:n])
plt.scatter(X[0,n:], X[1,n:])

# H1
w1, b1 = findWeight(normal=(0,1), coord_up=(0,0), coord_down=(-6,-3))
slope1 = normalToSlope(w1)
y1, y_up1, y_down1 = findEquation(x, w1, b1)

plt.plot(x, y1, 'k-')
plt.plot(x, y_up1, 'k--')
plt.plot(x, y_down1, 'k:')

# H2
w2, b2 = findWeight(normal=(-0.3,1), coord_up=(5,1), coord_down=(-6,-3))
slope2 = normalToSlope(w2)
y2, y_up2, y_down2 = findEquation(x, w2, b2)

plt.plot(x, y2, 'b-')
plt.plot(x, y_up2, 'b--')
plt.plot(x, y_down2, 'b:')

# H3
Wtilde = SVM4342()
Wtilde.fit(X.T, y)
b3 = Wtilde.b
w3 = Wtilde.w
slope3 = normalToSlope(w3)
y3, y_up3, y_down3 = findEquation(x, w3, b3)

plt.plot(x, y3, 'r-')
plt.plot(x, y_up3, 'r--')
plt.plot(x, y_down3, 'r:')
print(w1, w2, w3)
print(f"Hyperplane 1 margin: {2/np.linalg.norm(w1)}")
print(f"Hyperplane 2 margin: {2/np.linalg.norm(w2)}")
print(f"Hyperplane 3 margin: {2/np.linalg.norm(w3)}")
plt.show()
