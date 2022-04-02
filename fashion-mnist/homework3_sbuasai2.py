import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression(Xtilde, y, epsilon, batchSize, alpha):
    # initialize epoch and randomize weights
    epoch = 10
    Wtilde = 1e-5 * np.random.rand(Xtilde.shape[0],y.shape[1])
    # Compute stochastic gradient descent
    for _ in range(0, epoch):
        for i in range(0, (y.shape[0]//batchSize)):
            # initiliaze the starting and ending idx of the current batch
            start_idx = i*batchSize
            end_idx = start_idx+batchSize
            # compute the gradient and update the weights based on the current batch
            Wtilde -= epsilon*gradfCE(Xtilde[:,start_idx:end_idx], Wtilde, y[start_idx:end_idx,:], alpha)
    return Wtilde

# Given x data set of column vectors, yhat guesses, and y.
# Compute the gradient of fCE using formula: 1/n * x.dot((yhat-y))
def gradfCE (Xtilde, Wtilde, y, alpha=0):
    # initialize L2 regularization term
    w = np.copy(Wtilde)
    w[-1] = 0
    # initialize yhat from activation function yhat = softmax(z)
    z = Xtilde.T.dot(Wtilde)
    yhat = np.exp(z) / np.sum(np.exp(z), axis=1)[:,None]
    return (1/len(y)) * (Xtilde.dot((yhat - y)) + (alpha * w))

# Visualization helper
def vizWeights (weight):
    # construct 48x48 image from flatten image
    img = weight[:-1].reshape((28,28))
    # visualize the image
    plt.imshow(img, cmap='gray')
    plt.show()

def fPC (Xtilde, Wtilde, y):
    z = Xtilde.T.dot(Wtilde)
    yhat_arr = np.exp(z) / np.sum(np.exp(z), axis=1)[:,None]
    yhat = np.argmax(yhat_arr, axis=1)
    return np.mean(yhat == y)

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
    Wtilde = softmaxRegression(Xtilde_tr, Ytr, epsilon=0.1, batchSize=100, alpha=0.1)
    print(f"Training fPC: {fPC(Xtilde_tr, Wtilde, trainingLabels)}")
    print(f"Testing  fPC: {fPC(Xtilde_te, Wtilde, testingLabels)}")

    # Visualize the vectors
    for i in range(10):
        vizWeights(Wtilde[:,i])
