import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression(Xtilde, y, epsilon, batchSize, alpha):
    # initialize epoch
    epoch = 1
    batchSize = 2
    # randomize weights
    Wtilde = 0.01 * np.random.rand(Xtilde.shape[0],y.shape[0])
    # Compute stochastic gradient descent
    for _ in range(0, epoch):
        # for i in range(0, (y.shape[1]//batchSize)):
        for i in range(0, 1):
            # initiliaze the starting and ending idx of the current batch
            start_idx = i*batchSize
            end_idx = start_idx+batchSize
            # compute the gradient and update the weights based on the current batch
            gradient = gradfCE(Xtilde[:,start_idx:end_idx], Wtilde, y[:,start_idx:end_idx], alpha)
            Wtilde -= epsilon*gradient
    return Wtilde

# Given x data set of column vectors, yhat guesses, and y.
# Compute the gradient of fCE using formula: 1/n * x.dot((yhat-y))
def gradfCE (Xtilde, Wtilde, y, alpha=0):
    # initialize L2 regularization term
    w = np.copy(Wtilde)
    w[-1] = 0.5
    # initialize yhat from activation function yhat = softmax(z)
    z = Xtilde.T.dot(Wtilde)
    yhat = np.exp(z) / np.sum(np.exp(z), axis=1)[:,None]
    return (1/len(y)) * (Xtilde.dot((yhat - y.T)) + (alpha * w))

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    Xtilde_tr = np.column_stack((trainingImages, np.ones(trainingImages.shape[0]))).T
    Xtilde_te = np.column_stack((testingImages, np.ones(testingImages.shape[0]))).T

    # Change from 0-9 labels to "one-hot" binary vector labels. For instance, 
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    Ytr = np.zeros((10, trainingLabels.shape[0]))
    Ytr[trainingLabels, np.arange(trainingLabels.shape[0])] = 1
    Yte = np.zeros((10, testingLabels.shape[0]))
    Yte[testingLabels, np.arange(testingLabels.shape[0])] = 1

    # Train the model
    Wtilde = softmaxRegression(Xtilde_tr, Ytr, epsilon=0.1, batchSize=100, alpha=.1)

    # Visualize the vectors
    # ...
