import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
# ----------------------------------------------------
# Argument: Xtilde (trainingImages), y (trainingLabels) ,epsilon, batchSize, alpha, showfCE (bool to show last 20 fCE)
# Return: Wtilde
def softmaxRegression(Xtilde, y, epsilon, batchSize, alpha, showfCE=True):
    # initialize epoch and randomize weights
    epoch = 10
    Wtilde = 1e-5 * np.random.rand(Xtilde.shape[0],y.shape[1])
    # Compute stochastic gradient descent
    for e in range(0, epoch):
        for i in range(0, (y.shape[0]//batchSize)):
            # initiliaze the starting and ending idx of the current batch
            start_idx = i*batchSize
            end_idx = start_idx+batchSize
            # compute the gradient and update the weights based on the current batch
            Wtilde -= epsilon*gradfCE(Xtilde[:,start_idx:end_idx], Wtilde, y[start_idx:end_idx,:], alpha)
            # print out last 20 SDG batches
            if showfCE and e == 9 and ((y.shape[0]//batchSize) - i) <= 20:
                print(f"batch {i + 21 - (y.shape[0]//batchSize)} from last fCE: {fCE(Xtilde, Wtilde, y, alpha)}")
    return Wtilde

# Given x data set of column vectors, yhat guesses, and y.
# Compute the gradient of fCE using formula: 1/n * x.dot((yhat-y))
# ----------------------------------------------------
# Argument: Xtilde, Wtilde, y, alpha
# Return: gradient fCE of current Wtilde
def gradfCE (Xtilde, Wtilde, y, alpha=0.):
    # initialize L2 regularization term
    w = np.copy(Wtilde)
    w[-1] = 0
    yhat = softMax(Xtilde, Wtilde)
    # initialize yhat from activation function yhat = softmax(z)
    return (1/len(y)) * (Xtilde.dot((yhat - y)) + (alpha * w))

# Cross-entropy function
# ----------------------------------------------------
# Argument: Xtilde, Wtilde, y, alpha
# Return: fCE of given Wtilde
def fCE (Xtilde, Wtilde, y, alpha=0.):
    # initialize L2 regularized term
    reg = (alpha/(2*y.shape[0])) * np.trace(Wtilde[:-1,:].T.dot(Wtilde[:-1,:]))
    # initialize main fCE term
    main = -1 * np.mean(np.sum(y * np.log(softMax(Xtilde, Wtilde)), axis=1))
    return main + reg

# Percent correct function
# ----------------------------------------------------
# Argument: Xtilde, Wtilde, y, alpha
# Return: fCE of given Wtilde
def fPC (Xtilde, Wtilde, y):
    # initialize yhat
    yhat_arr = softMax(Xtilde, Wtilde)
    # compute yhat guesses into concrete result
    # eg: if yhat_arr = [0.6,0.2,0.2], yhat = [1,0,0]
    yhat = np.argmax(yhat_arr, axis=1)
    return np.mean(yhat == y)

# softMax helper to compute yhat
def softMax (Xtilde, Wtilde):
    z = Xtilde.T.dot(Wtilde)
    yhat = np.exp(z) / np.sum(np.exp(z), axis=1)[:,None]
    return yhat

# Visualization helper
def vizWeights (weight):
    # construct 48x48 image from flatten image
    img = weight[:-1].reshape((28,28))
    # visualize the image
    plt.imshow(img, cmap='gray')
    plt.show()

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
    Ytr = np.zeros((10, trainingLabels.shape[0])).T
    Ytr[np.arange(trainingLabels.shape[0]), trainingLabels] = 1
    Yte = np.zeros((10, testingLabels.shape[0])).T
    Yte[np.arange(testingLabels.shape[0]), testingLabels] = 1

    # Train the model
    Wtilde = softmaxRegression(Xtilde_tr, Ytr, epsilon=0.1, batchSize=100, alpha=0.1, showfCE=True)
    print(f"Training fPC: {fPC(Xtilde_tr, Wtilde, trainingLabels)}")
    print(f"Testing  fPC: {fPC(Xtilde_te, Wtilde, testingLabels)}")

    # Visualize the vectors
    # for i in range(10):
    #     vizWeights(Wtilde[:,i])
