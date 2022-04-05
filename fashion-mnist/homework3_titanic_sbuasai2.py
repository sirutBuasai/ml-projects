import pandas
import numpy as np

# Softmax regression function
# ----------------------------------------------------
# Argument: Xtilde , y  ,epsilon, batchSize, alpha
# Return: Wtilde
def softmaxRegression(Xtilde, y, epsilon, batchSize, alpha):
    # initialize epoch and randomize weights
    epoch = 10
    Wtilde = 1e-3 * np.random.rand(Xtilde.shape[0],y.shape[1])
    # Compute stochastic gradient descent
    for _ in range(0, epoch):
        for i in range(0, (y.shape[0]//batchSize)):
            # initiliaze the starting and ending idx of the current batch
            start_idx = i*batchSize
            end_idx = start_idx+batchSize
            # compute the gradient and update the weights based on the current batch
            Wtilde -= epsilon*gradfCE(Xtilde[:,start_idx:end_idx], Wtilde, y[start_idx:end_idx,:], alpha)
    return Wtilde

# Gradient of cross-entropy function
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

if __name__ == "__main__":
    # Load training data
    dtr = pandas.read_csv("train.csv")
    ytr = dtr.Survived.to_numpy()
    sex_tr = dtr.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass_tr = dtr.Pclass.to_numpy()
    SibSp_tr = dtr.SibSp.to_numpy()

    # Stack each attributes onto the one matrix
    Xtilde_tr = np.vstack((sex_tr, Pclass_tr, SibSp_tr, np.ones(ytr.shape[0])))
    Ytr = np.zeros((2, ytr.shape[0])).T
    Ytr[np.arange(ytr.shape[0]), ytr] = 1

    # Train model using part of homework 3.
    Wtilde = softmaxRegression(Xtilde_tr, Ytr, epsilon=1, batchSize=100, alpha=1)

    # Load testing data
    dte = pandas.read_csv("test.csv")
    PassengerId_te = dte.PassengerId.to_numpy()
    sex_te = dte.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass_te = dte.Pclass.to_numpy()
    SibSp_te = dte.SibSp.to_numpy()

    # Stack each attributes onto the one matrix
    Xtilde_te = np.vstack((sex_te, Pclass_te, SibSp_te, np.ones(sex_te.shape[0])))

    # Compute predictions on test set
    print(f"Training fPC: {fPC(Xtilde_tr, Wtilde, ytr)}")
    yhat_te = np.argmax(softMax(Xtilde_te, Wtilde), axis=1)

    # Write CSV file of the format:
    # PassengerId, Survived
    df = pandas.DataFrame({'PassengerId': PassengerId_te,'Survived': yhat_te})
    df.to_csv("result.csv", index=False)
