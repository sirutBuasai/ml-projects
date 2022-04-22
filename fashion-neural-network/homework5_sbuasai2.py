import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import random

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.


def unpack(w):
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
    return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack(W1, b1, W2, b2):
    return np.hstack((W1.flatten(), b1, W2.flatten(), b2))

# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("fashion_mnist_{}_images.npy".format(which)) / 255.
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))

    # Convert labels vector to one-hot matrix (N x C).
    labelsTilde = np.zeros((10, labels.shape[0])).T
    labelsTilde[np.arange(labels.shape[0]), labels] = 1

    return images, labelsTilde

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss, accuracy,
# as well as the intermediate values of the NN.
def fCE(X, Y, w, alpha=0.): #CHECK
    W1, b1, W2, b2 = unpack(w)
    # get z1,h1,yhat from forward propagation
    z1, h1, z2, yhat = forwardProp(X, w)
    # get percent correct accuracy
    acc = fPC(yhat.T, Y)
    # calculate regularlized fCE
    cost = -1 * np.mean(np.sum(Y * np.log(yhat.T), axis=1))

    return cost, acc, z1, h1, W1, W2, yhat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, Y, w, alpha=0.): #CHECK
    W1, b1, W2, b2 = unpack(w)
    z1, h1, z2, yhat = forwardProp(X, w)

    db2 = np.mean(yhat.T - Y, axis=0)
    dW2 = np.atleast_2d(yhat.T - Y).T.dot(h1).T
    g = ((yhat.T - Y).dot(W2)) * ReLUPrime(z1)
    db1 = np.mean(g, axis=0)
    dW1 = X.T.dot(np.atleast_2d(g))

    grad = pack(dW1.T, db1, dW2.T, db2)

    return grad

#
# Helper functions ########################################
# ReLU: Given an array z, return the ReLU(z)
# ---------------------------------------------------------
# Argument: z
# Return: z
def ReLU(z): #CHECK
    z[z <= 0] = 0
    return z

# ReLUPrime: given an array z, return the derivative of z
# ---------------------------------------------------------
# Argument: z
# Return: z
def ReLUPrime(z): #CHECK
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

# Percent correct function: given Xtilde, Wtilde, calculate yhat
# and return percent correct wrt to y
# ----------------------------------------------------
# Argument: yhat, y
# Return: fPC
def fPC(guess, ground_truth):
    # compute yhat guesses into concrete result
    # eg: yhat = [0.6,0.2,0.2] -> yhat = [1,0,0]
    yhat = np.array([np.argmax(i) for i in guess])
    y =np.array([np.argmax(i) for i in ground_truth])

    return np.mean(yhat == y)

# SoftMax helper funciton to compute yhat
# ----------------------------------------------------
# Argument: z
# Return: yhat
def softMax(z): #CHECK
    num = np.exp(z).T
    denom = np.atleast_2d(np.exp(z)).sum(axis=1)
    yhat = num / denom
    return yhat

# Forward propagation helper funciton to z1, h1, z2, yhat
# ----------------------------------------------------
# Argument: X, w
# Return: z1, h1, z2, yhat
def forwardProp(X, w): #CHECK
    W1, b1, W2, b2 = unpack(w)

    z1 = X.dot(W1.T) + b1
    h1 = ReLU(z1)
    z2 = h1.dot(W2.T) + b2
    yhat = softMax(z2)

    return z1, h1, z2, yhat

# Visualization helper
# ----------------------------------------------------
# Argument: x
# Return: void
def visualize(x):
    # construct 48x48 image from flatten image
    img = x.reshape((28, 28))
    # visualize the image
    plt.imshow(img, cmap='gray')
    plt.show()

def findBestHyperparameters(trainX, trainY, testX, testY, w):
    epochTrain = [10, 20, 30, 40, 50, 75, 100]
    batchSizeTrain = [16, 32, 64, 128, 256]
    learningRateTrain = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    alphaTrain = [0.001]

    optimal = {'Epoch': 0, 'Batch': 0, 'learnRate': 0.0, 'Cost': float('inf'), 'Accuracy': 0.0, 'Alpha': 1e-3}

    for x in range(1,10):
        epochs = random.choice(epochTrain)
        learn_rate = random.choice(learningRateTrain)
        batch = random.choice(batchSizeTrain)
        alpha = random.choice(alphaTrain)
        cost, acc = train(trainX, trainY, testX, testY, w, epochs, batch, learn_rate, alpha, test=False)
        print("Hyper Parameters in run ",x, "\nEpochs: ", epochs, "\nLearning Rate: ", learn_rate, "\nBatch Size: ", batch)

        if cost < optimal.get('Cost') and acc > optimal.get('Accuracy'):
            optimal.update(Epoch = epochs, Batch = batch, learnRate = learn_rate, Cost = cost, Accuracy = acc)

    print(optimal)

    return optimal.get('Epoch'), optimal.get('Batch'), optimal.get('learnRate'), optimal.get('Alpha')

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train(trainX, trainY, testX, testY, w, epochs, batch, learn_rate, alpha, test=False):
    EPOCH = epochs
    BATCH_SIZE = batch
    LEARNING_RATE = learn_rate
    ALPHA = alpha
    W1, b1, W2, b2 = unpack(w)

    for e in range(EPOCH):
        # randomize the samples
        rand_idx = np.random.permutation(trainX.shape[0])
        randX = trainX[rand_idx,:]
        randY = trainY[rand_idx,:]
        # process the epoch batch by batch
        for i in range(0, (trainY.shape[0]//BATCH_SIZE)):
            # initialize the starting and ending index of the current batch
            start_idx = i*BATCH_SIZE
            end_idx = start_idx+BATCH_SIZE
            batchX = randX[start_idx:end_idx,:]
            batchY = randY[start_idx:end_idx,:]
            # compute the gradient and update the weights based on the current batch
            grad = w * ALPHA + gradCE(batchX, batchY, w) * (1 - ALPHA)
            w -= LEARNING_RATE * grad


    if test:
        cost, acc, _, _, _, _, _ = fCE(testX, testY, w, alpha)
    else:
        cost, acc, _, _, _, _, _ = fCE(trainX, trainY, w, alpha)
    print("Cost:", cost)
    print("Accuracy:", acc)
    return cost, acc

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    # idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    # print("Numerical gradient:")
    # print(scipy.optimize.approx_fprime(w, lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_)[0], 1e-10))
    # print("Analytical gradient:")
    # print(gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w))
    # print("Discrepancy:")
    # print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_)[0], \
    #                                 lambda w_: gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_), \
    #                                 w))

    # # # # Train the network using SGD.
    e,b,l,a = findBestHyperparameters(trainX, trainY, testX, testY, w)
    # e,b,l,a = 100, 64, 0.001, 0.01
    train(trainX, trainY, testX, testY, w, e, b, l, a, test=True)
