# external libraries
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
# internal libraries
from parsing import *

# Train and thest the data using k-nearest neighbor
def kNearestNeighbor(trainX, trainY, testX):
    # hyperparameters
    NUM_NEIGHBOR = 7

    # train the model using k-nearest neighbor
    model = KNeighborsClassifier(n_neighbors=NUM_NEIGHBOR)
    model.fit(trainX, trainY)

    # use weights on testing set
    guess = model.predict(testX)
    predictions = decodeOHE(guess)

    return predictions

# Train and test the data using 3-layers neural network
# TODO: add validation on hyperparameters
def threeLayerNN(trainX, trainY, validX, vliadY, testX):
    # hyperparameters
    EPOCH = 20
    LEARN_RATE = 0.001
    NUM_HIDDEN = 50

    # initialize neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(NUM_HIDDEN, activation=tf.nn.relu, name="layer1"))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="layer2"))

    # train the neural network
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(trainX, trainY, epochs=EPOCH)

    # use weights on testing set
    guess = model.predict(testX)
    predictions = decodeOHE(guess)

    return predictions

# Train and test the data using deep neural network
# TODO: add validation on hyperparameters
def deepNN(trainX, trainY, validX, vliadY, testX):
    # hyperparameters
    EPOCH = 20
    LEARN_RATE = 0.001
    NUM_HIDDEN1 = 50
    NUM_HIDDEN2 = 50
    NUM_HIDDEN3 = 50

    # initialize neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(NUM_HIDDEN1, activation=tf.nn.relu, name="layer1"))
    model.add(tf.keras.layers.Dense(NUM_HIDDEN2, activation=tf.nn.relu, name="layer2"))
    model.add(tf.keras.layers.Dense(NUM_HIDDEN3, activation=tf.nn.relu, name="layer3"))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="layer4"))

    # train the neural network
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(trainX, trainY, epochs=EPOCH)

    # use weights on testing set
    guess = model.predict(testX)
    predictions = decodeOHE(guess)

    return predictions
