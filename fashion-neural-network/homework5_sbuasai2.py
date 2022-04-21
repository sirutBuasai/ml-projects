import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    # Unpack arguments
    start = 0
    end = NUM_HIDDEN*NUM_INPUT
    W1 = w[0:end]
    start = end
    end = end + NUM_HIDDEN
    b1 = w[start:end]
    start = end
    end = end + NUM_OUTPUT*NUM_HIDDEN
    W2 = w[start:end]
    start = end
    end = end + NUM_OUTPUT
    b2 = w[start:end]
    # Convert from vectors into matrices
    W1 = W1.reshape(NUM_HIDDEN, NUM_INPUT)
    W2 = W2.reshape(NUM_OUTPUT, NUM_HIDDEN)
    return W1,b1,W2,b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    return np.hstack((W1.flatten(), b1, W2.flatten(), b2))

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("fashion_mnist_{}_images.npy".format(which)).T / 255.
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))

    # Convert labels vector to one-hot matrix (C x N).
    labelsTilde = np.zeros((10, labels.shape[0])).T
    labelsTilde[np.arange(labels.shape[0]), labels] = 1

    return images, labelsTilde.T

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss, accuracy,
# as well as the intermediate values of the NN.
def fCE (X, Y, w, alpha=0.):
    W1, b1, W2, b2 = unpack(w)
    # get z1,h1,yhat from forward propagation
    z1,h1,yhat = forwardProp(X,w)
    acc = fPC(yhat, Y)
    # calculate regularlized fCE
    reg = (alpha*(2*Y.shape[1])) * (np.sum(W1**2)+np.sum(W2**2))
    main = -1 * np.mean(np.sum(Y.T * np.log(yhat).T, axis=1))
    cost = main + reg

    return cost, acc, z1, h1, W1, W2, yhat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    z1,h1,yhat = forwardProp(X,w)

    db2 = np.mean(yhat - Y, axis=0)
    dW2 = (yhat - Y).dot(h1.T)
    g = (yhat - Y).T.dot(W2).T * ReLUPrime(z1)
    db1 = np.mean(g, axis=0)
    dW1 = X.dot(g.T)

    grad = pack(dW1, db1, dW2, db2)

    return grad

#
# Helper functions ########################################
# ReLU: Given an array z, return the ReLU(z)
# ---------------------------------------------------------
# Argument: z
# Return: z
def ReLU (z):
    z[z<=0] = 0
    return z

# ReLUPrime: given an array z, return the derivative of z
# ---------------------------------------------------------
# Argument: z
# Return: z
def ReLUPrime (z):
    z[z<=0] = 0
    z[z>1] = 1
    return z

# Percent correct function: given Xtilde, Wtilde, calculate yhat
# and return percent correct wrt to y
# ----------------------------------------------------
# Argument: Xtilde, Wtilde, y, alpha
# Return: fPC
def fPC (yhat, y):
    # compute yhat guesses into concrete result
    # eg: yhat = [0.6,0.2,0.2] -> yhat = [1,0,0]
    max_idx = np.argmax(yhat, axis=0)
    yhat = np.zeros((10, max_idx.shape[0]))
    yhat[max_idx, np.arange(max_idx.shape[0])] = 1
    return np.mean(yhat == y)

# SoftMax helper funciton to compute yhat
# ----------------------------------------------------
# Argument: z
# Return: yhat
def softMax (z):
    yhat = np.exp(z) / np.sum(np.exp(z), axis=1)[:,None]
    return yhat

# Forward propagation helper funciton to z1, h1, z2, yhat
# ----------------------------------------------------
# Argument: z
# Return: yhat
def forwardProp (X, w):
    W1,b1,W2,b2 = unpack(w)
    z1 = W1.dot(X) + b1[:,None]
    h1 = ReLU(z1)
    z2 = W2.dot(h1) + b2[:,None]
    yhat = softMax(z2)
    return z1,h1,yhat

# Visualization helper
# ----------------------------------------------------
# Argument: x
# Return: void
def visualize (x):
    # construct 48x48 image from flatten image
    img = x.reshape((28,28))
    # visualize the image
    plt.imshow(img, cmap='gray')
    plt.show()

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX, trainY, testX, testY, w):
    EPOCH = 10
    BATCH_SIZE = 1
    LEARNING_RATE = 3e-3
    ALPHA = 1e-3
    b = 0.1
    W1,b1,W2,b2 = unpack(w)

    for e in range(EPOCH):
        # randomize the samples
        rand_idx = np.random.permutation(trainX.shape[1])
        randX = trainX[:,rand_idx]
        randY = trainY[:,rand_idx]
        # process the epoch batch by batch
        for i in range(0, (trainY.shape[1]//BATCH_SIZE)):
            # initialize the starting and ending index of the current batch
            start_idx = i*BATCH_SIZE
            end_idx = start_idx+BATCH_SIZE
            batchX = randX[:,start_idx:end_idx]
            batchY = randY[:,start_idx:end_idx]
            # compute the gradient and update the weights based on the current batch
            grad = w * b + gradCE(batchX, batchY, w) * (1 - b)
            w = w - (LEARNING_RATE * grad)

    cost, acc, _,_,_,_,_ = fCE(trainX, trainY, w, ALPHA)
    print("Loss:", cost)
    print("Accuracy:", acc)

if __name__ == "__main__":
    # Load data
    # if "trainX" not in globals():
    #     trainX, trainY = loadData("train")
    #     testX, testY = loadData("test")
    trainX, trainY = loadData("train")
    testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)
    alpha = 1e-3
    fCE(trainX, trainY, w, alpha)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    # idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    # print("Numerical gradient:")
    # print(scipy.optimize.approx_fprime(w, lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_, alpha)[0], 1e-10))
    # print("Analytical gradient:")
    # print(gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w))
    # print("Discrepancy:")
    # print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_, alpha)[0], \
    #                                 lambda w_: gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_), \
    #                                 w))

    # # Train the network using SGD.
    # train(trainX, trainY, testX, testY, w)
