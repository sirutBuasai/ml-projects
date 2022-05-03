# external libraries
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# internal libraries
from parsing import *

# Guess the data randomly
def randomGuess(trainX, trainY, testX):
    # Given that Ghoul class is most dominant in training set
    # find the accuray from the training set
    yhat = np.tile(np.array([0,1,0]), (trainX.shape[0], 1))
    accuracy = accuracy_score(y_true=trainY.to_numpy(), y_pred=yhat)
    print("Random guess training set accuracy:", accuracy)
    # construct the guess for the testing set
    guess = np.tile(np.array([0,1,0]), (testX.shape[0], 1))
    predictions = decodeOHE(guess)
    return predictions

# Train and test the data using k-nearest neighbor
def kNearestNeighbor(trainX, trainY, testX):
    # hyperparameters
    NUM_NEIGHBOR = 7

    # train the model using k-nearest neighbor
    model = KNeighborsClassifier(n_neighbors=NUM_NEIGHBOR)
    model.fit(trainX, trainY)

    # use weights on testing set
    print("K-NearestNeighbor training accuracy:", model.score(trainX, trainY))
    guess = model.predict(testX)
    predictions = decodeOHE(guess)

    return predictions

# Train and test the data using 3-layers neural network
def threeLayerNN(trainX, trainY, testX):
    # hyperparameters
    EPOCH = 100
    BATCH_SIZE = 64
    LEARN_RATE = 0.001
    NUM_HIDDEN = 50

    # initialize neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(NUM_HIDDEN, activation=tf.nn.relu, name="hidden"))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="output"))

    # train the neural network
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE)

    # use weights on testing set
    guess = model.predict(testX)
    predictions = decodeOHE(guess)

    return predictions

# Train and test the data using deep neural network
def deepNN(trainX, trainY, testX):
    # hyperparameters
    EPOCH = 100
    BATCH_SIZE = 64
    ALPHA = 0.1
    LEARN_RATE = 0.005
    NUM_HIDDEN1 = 50
    NUM_HIDDEN2 = 50
    NUM_HIDDEN3 = 50

    # initialize neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(NUM_HIDDEN1, activation=tf.nn.relu, name="hidden1"))
    model.add(tf.keras.layers.Dense(NUM_HIDDEN2, activation=tf.nn.relu, name="hidden2"))
    model.add(tf.keras.layers.Dropout(rate=ALPHA))
    model.add(tf.keras.layers.Dense(NUM_HIDDEN3, activation=tf.nn.relu, name="hidden3"))
    model.add(tf.keras.layers.Dense(NUM_HIDDEN3, activation=tf.nn.relu, name="hidden4"))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="hidden5"))

    # train the neural network
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE)

    # use weights on testing set
    guess = model.predict(testX)
    predictions = decodeOHE(guess)

    return predictions
