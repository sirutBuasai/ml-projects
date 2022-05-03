# external libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
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
    print("Baseline K-NearestNeighbor training accuracy:", model.score(trainX, trainY))
    guess = model.predict(testX)
    predictions = decodeOHE(guess)

    return predictions

def optimizeKNN(trainX, trainY, testX):

    # Doing another model with hyperparameter tuning
    model = KNeighborsClassifier(algorithm='auto')
    # Params to be tested
    gridsearch_params = {
        'n_neighbors': (5, 7, 9),
        'leaf_size': (5, 10, 20),}

    gridsearch_knn = GridSearchCV(
        estimator=model,
        param_grid=gridsearch_params,
        scoring='accuracy',
        n_jobs=-1,
        cv=None
    )

    knn = gridsearch_knn.fit(trainX, trainY)
    guess = knn.predict(testX)
    predictions = decodeOHE(guess)

    print("Best params for knn:", gridsearch_knn.best_params_)
    print("Accuracy on training for knn tuned:", knn.score(trainX, trainY))
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

def optimize3NN(trainX, trainY, validX, validY, testX):
    EPOCHS = np.array([25, 50, 100])
    BATCH_SIZES = np.array([16, 32, 64, 128])
    # NUM_HIDDENS = np.array([30, 40, 50])
    NUM_HIDDENS = np.array([80, 100, 120])
    LEARN_RATES = np.array([0.01, 0.001])

    # initialize a dictionary to store the optimal hyperparameters
    best_hp = {
        'hidden': 0.0,
        'batch': 0.0,
        'learn_rate': 0.0,
        'epoch': 0.0,
        'accuracy': 0.0
    }

    # Constant for number of hyperparameter combos tested
    COUNT = 20

    for _ in range(COUNT):
        # randomly choose a set of hyperparameters
        NUM_HIDDEN = np.random.choice(NUM_HIDDENS)
        BATCH_SIZE = np.random.choice(BATCH_SIZES)
        LEARN_RATE = np.random.choice(LEARN_RATES)
        EPOCH = np.random.choice(EPOCHS)

        print("Hidden layers:", NUM_HIDDEN)
        print("Batch size:", BATCH_SIZE)
        print("Learn rate:", LEARN_RATE)
        print("Epoch:", EPOCH)

        # initialize neural network
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(NUM_HIDDEN, activation=tf.nn.relu, name="hidden"))
        model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="output"))

        # train the neural network
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        model.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE)

        # Getting loss and accuracy on validation data
        score = model.evaluate(validX, validY)
        acc = score[1]
        print("----------------------------")
        print("Accuracy")

        if acc > best_hp['accuracy']:
            best_hp.update(hidden=NUM_HIDDEN,
                           batch=BATCH_SIZE,
                           learn_rate=LEARN_RATE,
                           epoch=EPOCH,
                           accuracy=acc)


    # Training new NN to get accuracy and produce results
    model_hp = tf.keras.Sequential()
    model_hp.add(tf.keras.layers.Dense(best_hp['hidden'], activation=tf.nn.relu, name="hidden"))
    model_hp.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="output"))

    # train the neural network
    model_hp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_hp['learn_rate']),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model_hp.fit(trainX, trainY, epochs=best_hp['epoch'], batch_size=best_hp['batch'])

    # use weights on testing set
    guess = model_hp.predict(testX)
    predictions = decodeOHE(guess)

    print("Final accuracy tuned 3 layer:", best_hp['accuracy'])
    print("Best params:", best_hp)

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

def optimizeDNN(trainX, trainY, validX, validY, testX):
    EPOCHS = np.array([25, 50, 100])
    BATCH_SIZES = np.array([16, 32, 64, 128])
    # NUM_HIDDENS1 = np.array([40, 50, 60])
    NUM_HIDDENS1 = np.array([80, 100, 120])
    # NUM_HIDDENS2 = np.array([40, 50, 60])
    NUM_HIDDENS2 = np.array([80, 100, 120])
    # NUM_HIDDENS3 = np.array([40, 50, 60])
    NUM_HIDDENS3 = np.array([80, 100, 120])
    LEARN_RATES = np.array([0.01, 0.001])
    ALPHA = 0.1

    # initialize a dictionary to store the optimal hyperparameters
    best_hp = {
        'hidden1': 0.0,
        'hidden2': 0.0,
        'hidden3': 0.0,
        'batch': 0.0,
        'learn_rate': 0.0,
        'epoch': 0.0,
        'accuracy': 0.0
    }

    # Constant for number of hyperparameter combos tested
    COUNT = 20

    for _ in range(COUNT):
        # randomly choose a set of hyperparameters
        NUM_HIDDEN1 = np.random.choice(NUM_HIDDENS1)
        NUM_HIDDEN2 = np.random.choice(NUM_HIDDENS2)
        NUM_HIDDEN3 = np.random.choice(NUM_HIDDENS3)
        BATCH_SIZE = np.random.choice(BATCH_SIZES)
        LEARN_RATE = np.random.choice(LEARN_RATES)
        EPOCH = np.random.choice(EPOCHS)

        print("Hidden layers 1:", NUM_HIDDEN1)
        print("Hidden layers 2:", NUM_HIDDEN2)
        print("Hidden layers 3:", NUM_HIDDEN3)
        print("Batch size:", BATCH_SIZE)
        print("Learn rate:", LEARN_RATE)
        print("Epoch:", EPOCH)

        # initialize neural network
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(NUM_HIDDEN1, activation=tf.nn.relu, name="hidden1"))
        model.add(tf.keras.layers.Dense(NUM_HIDDEN2, activation=tf.nn.relu, name="hidden2"))
        model.add(tf.keras.layers.Dropout(rate=ALPHA))
        model.add(tf.keras.layers.Dense(NUM_HIDDEN3, activation=tf.nn.relu, name="hidden3"))
        model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="output"))

        # train the neural network
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        model.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE)

        # Getting loss and accuracy on validation data
        score = model.evaluate(validX, validY)
        acc = score[1]
        print("----------------------------")
        print("Accuracy")

        if acc > best_hp['accuracy']:
            best_hp.update(hidden1=NUM_HIDDEN1,
                           hidden2=NUM_HIDDEN2,
                           hidden3=NUM_HIDDEN3,
                           batch=BATCH_SIZE,
                           learn_rate=LEARN_RATE,
                           epoch=EPOCH,
                           accuracy=acc)

    print("Best params:", best_hp)
    # Using tuned params to generate output
    # initialize neural network
    model_hp = tf.keras.Sequential()
    model_hp.add(tf.keras.layers.Dense(best_hp['hidden1'], activation=tf.nn.relu, name="hidden1"))
    model_hp.add(tf.keras.layers.Dense(best_hp['hidden2'], activation=tf.nn.relu, name="hidden2"))
    model_hp.add(tf.keras.layers.Dropout(rate=ALPHA))
    model_hp.add(tf.keras.layers.Dense(best_hp['hidden3'], activation=tf.nn.relu, name="hidden3"))
    model_hp.add(tf.keras.layers.Dense(best_hp['hidden3'], activation=tf.nn.relu, name="hidden4"))
    model_hp.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="hidden5"))

    # train the neural network
    model_hp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_hp['learn_rate']),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model_hp.fit(trainX, trainY, epochs=best_hp['epoch'], batch_size=best_hp['batch'])

    # use weights on testing set
    guess = model_hp.predict(testX)
    predictions = decodeOHE(guess)

    return predictions
