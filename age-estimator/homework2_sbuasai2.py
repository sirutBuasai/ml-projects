import numpy as np
from matplotlib import pyplot as plt

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    pass

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    # flatten the image
    flatten_faces = np.reshape(faces, (faces.shape[0], faces.shape[1]*faces.shape[2]))
    # append 1s to each row
    faces_tilde = np.column_stack((flatten_faces, np.ones(flatten_faces.shape[0])))
    return faces_tilde

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (wtilde, Xtilde, y):
    # the formula is X.T.dot(w). however, the given Xtilde is already in ((M**2 + 1) x N) shape
    yhat = Xtilde.dot(wtilde)
    return np.mean((y-yhat)**2)

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (wtilde, Xtilde, y, alpha = 0.):
    # the formula is X.dot(X.T.dot(w)). however, the given Xtilde is already in ((M**2 + 1) x N) shape
    return (1/len(y)) * Xtilde.T.dot(Xtilde.dot(wtilde) - y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    # the formula is w = solve(X.dot(X.T), X.dot(y)). however, the given Xtilde is already in ((M**2 + 1) x N) shape
    # thus, varialbe X in the formula = X.T and vice versa
    wtilde = np.linalg.solve(Xtilde.T.dot(Xtilde), Xtilde.T.dot(y))
    return wtilde

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    # use gradient descent to get wtilde
    wtilde = gradientDescent(Xtilde, y)
    return wtilde

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    pass

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    # initialize wtilde
    wtilde = 0.01 * np.random.rand(Xtilde.shape[1])
    # iterate {T} times for gradient descent
    for i in range(T):
        # formula: w(1) = w(0) - epsilon * gradfMSE(w(0))
        wtilde -= EPSILON*gradfMSE(wtilde, Xtilde, y, alpha)
    return wtilde

# Visualization helper
def vizWeights(weight):
    # construct 48x48 image from flatten image
    img = weight[:-1].reshape((48,48))
    # visualize the image
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)
    # Report fMSE cost using each of the three learned weight vectors
    print(f"fMSE of training set with w1: {fMSE(w1, Xtilde_tr, ytr)}")
    print(f"fMSE of training set with w2: {fMSE(w2, Xtilde_tr, ytr)}")
    print(f"fMSE of testing  set with w1: {fMSE(w1, Xtilde_te, yte)}")
    print(f"fMSE of testing  set with w2: {fMSE(w2, Xtilde_te, yte)}")
    vizWeights(w1)
    vizWeights(w2)
