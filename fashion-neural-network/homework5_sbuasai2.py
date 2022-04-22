import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

EPOCH = 10  # Number of epochs
BATCH_SIZE = 16  # Size of SGD batch
LEARNING_RATE = 0.05   # Gradient descent rate
ALPHA = 0.001   # Regularization strength
NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 1  # Number of examples on which to check the gradient

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
    images = np.load("fashion_mnist_{}_images.npy".format(which)).T / 255.
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))

    # Convert labels vector to one-hot matrix (C x N).
    labelsTilde = np.zeros((10, labels.shape[0])).T
    labelsTilde[np.arange(labels.shape[0]), labels] = 1

    return images, labelsTilde.T

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss, accuracy,
# as well as the intermediate values of the NN.
def fCE(X, Y, w, alpha=0.):
    W1, b1, W2, b2 = unpack(w)
    # get z1,h1,yhat from forward propagation
    z1, h1, z2, yhat = forwardProp(X, w)
    # get percent correct accuracy
    acc = fPC(yhat.T, Y.T)
    # calculate regularlized fCE
    reg = (alpha/2) * (np.sum(W1**2) + np.sum(W2**2))
    main = -1 * np.mean(np.sum(Y * np.log(yhat), axis=0))
    cost = main + reg

    return cost, acc, z1, h1, W1, W2, yhat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, Y, w, alpha=0.):
    W1, b1, W2, b2 = unpack(w)
    z1, h1, z2, yhat = forwardProp(X, w)

    g = ((yhat - Y).T.dot(W2) * ReLUPrime(z1)).T
    dW1 = (np.atleast_2d(g).dot(X.T) / X.shape[1]) + (alpha*W1)
    db1 = np.mean(g, axis=1)
    dW2 = (np.atleast_2d(yhat - Y).dot(h1) / X.shape[1]) + (alpha*W2)
    db2 = np.mean(yhat - Y, axis=1)

    grad = pack(dW1, db1, dW2, db2)

    return grad

#
# Helper functions ########################################
# ReLU: Given an array z, return the ReLU(z)
# ---------------------------------------------------------
# Argument: z
# Return: z
def ReLU(z):
    z[z <= 0] = 0
    return z

# ReLUPrime: given an array z, return the derivative of z
# ---------------------------------------------------------
# Argument: z
# Return: z
def ReLUPrime(z):
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
    yhat = np.argmax(guess, axis=1)
    y = np.argmax(ground_truth, axis=1)

    return np.mean(yhat == y)

# SoftMax helper funciton to compute yhat
# ----------------------------------------------------
# Argument: z
# Return: yhat
def softMax(z):
    num = np.exp(z).T
    denom = np.atleast_2d(np.exp(z)).sum(axis=1)
    yhat = num / denom
    return yhat

# Forward propagation helper funciton to z1, h1, z2, yhat
# ----------------------------------------------------
# Argument: X, w
# Return: z1, h1, z2, yhat
def forwardProp(X, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = W1.dot(X).T + b1
    h1 = ReLU(z1)
    z2 = W2.dot(h1.T).T + b2
    yhat = softMax(z2)

    return z1, h1, z2, yhat

# Randomly initiialize weights helper
# ----------------------------------------------------
# Argument: void
# Return: w
def initWeights():
    # randomly initialize the weights
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    w = pack(W1, b1, W2, b2)

    return w

# Find the best hyper parameters using the validation set
# ----------------------------------------------------
# Argument: x
# Return: void
def findBestHyperparameters(trainX, trainY, testX, testY, count):
    # set global variables
    global NUM_HIDDEN, BATCH_SIZE, LEARNING_RATE, EPOCH, ALPHA

    # initialize the possible hyperparameters
    hidden_layer_list = np.array([30, 40, 50])
    batch_list = np.array([16, 32, 64, 128, 256])
    learn_rate_list = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
    epoch_list = np.array([10, 25, 50, 75, 100])
    reg_list = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
    # initialize a dictionary to store the optimal hyperparameters
    best_hp = {
        'hidden': 0.0,
        'batch': 0.0,
        'learn_rate': 0.0,
        'epoch': 0.0,
        'reg_str': 0.0,
        'cost': float('inf'),
        'accuracy': 0.0
    }

    for i in range(count):
        # randomly choose a set of hyperparameters
        NUM_HIDDEN = np.random.choice(hidden_layer_list)
        BATCH_SIZE = np.random.choice(batch_list)
        LEARNING_RATE = np.random.choice(learn_rate_list)
        EPOCH = np.random.choice(epoch_list)
        ALPHA = np.random.choice(reg_list)

        # randomly initialize the weights
        W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
        b1 = 0.01 * np.ones(NUM_HIDDEN)
        W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        b2 = 0.01 * np.ones(NUM_OUTPUT)
        w = pack(W1, b1, W2, b2)

        # train the model using given hyperparameters
        print("Run number", i+1)
        print("Training hyperparameters:")
        print("Hidden layers:", NUM_HIDDEN)
        print("Batch size:", BATCH_SIZE)
        print("Learning rate:", LEARNING_RATE)
        print("Epoch:", EPOCH)
        print("Regularization strength:", ALPHA)
        cost, acc = train(trainX, trainY, testX, testY, w, EPOCH, BATCH_SIZE, LEARNING_RATE, ALPHA, test=False)
        print("----------------------------------")

        # update the optimal hyperparameters if the cost is lower and the accuracy is higher
        if (cost < best_hp['cost']) and (acc > best_hp['accuracy']):
            best_hp.update(hidden = NUM_HIDDEN,
                           batch = BATCH_SIZE,
                           learn_rate = LEARNING_RATE,
                           epoch = EPOCH,
                           reg_str = ALPHA,
                           cost = cost,
                           accuracy = acc)

    print(best_hp)
    return best_hp['hidden'], best_hp['batch'], best_hp['learn_rate'], best_hp['epoch'], best_hp['reg_str']

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train(trainX, trainY, testX, testY, w, epochs, batch, learn_rate, alpha, test=False):
    W1, b1, W2, b2 = unpack(w)

    for e in range(epochs):
        # randomize the samples
        rand_idx = np.random.permutation(trainX.shape[1])
        randX = trainX[:,rand_idx]
        randY = trainY[:,rand_idx]
        # process the epoch batch by batch
        for i in range(0, (trainY.shape[1]//batch)):
            # initialize the starting and ending index of the current batch
            start_idx = i*batch
            end_idx = start_idx+batch
            batchX = randX[:,start_idx:end_idx]
            batchY = randY[:,start_idx:end_idx]
            # compute the gradient and update the weights based on the current batch
            w -= learn_rate * gradCE(batchX, batchY, w, alpha)
        # print epoch and losses
        if test and (epochs - e <= 20):
            cost, acc, _, _, _, _, _ = fCE(testX, testY, w, alpha)
            print("Epoch:", e+1, "Cost:", cost, "Accuracy:", acc)


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

    # Find the best hyper parameters
    NUM_HIDDEN, BATCH_SIZE, LEARNING_RATE, EPOCH, ALPHA = findBestHyperparameters(trainX, trainY, testX, testY, 20)

    # Initialize weights randomly
    w = initWeights()

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print("Numerical gradient:")
    print(scipy.optimize.approx_fprime(w, lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_, ALPHA)[0], 1e-10))
    print("Analytical gradient:")
    print(gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w, ALPHA))
    print("Discrepancy:")
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_, ALPHA)[0], \
                                    lambda w_: gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_, ALPHA), \
                                    w))

    # # # # Train the network using SGD.
    train(trainX, trainY, testX, testY, w, EPOCH, BATCH_SIZE, LEARNING_RATE, ALPHA, test=True)
