# external libraries
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# internal libraries
from parsing import *
from model import *

if __name__ == "__main__":
    # read data
    trainX, trainY = loadTrain("train")
    testX = loadTest("test")

    # permutate training and testing data
    rand = np.random.randint(1000)
    trainX = shuffle(trainX, random_state=rand)
    trainY = shuffle(trainY, random_state=rand)
    testX = shuffle(testX)

    # construct validation set
    rand = np.random.randint(1000)
    trainX, validX = train_test_split(trainX, test_size=0.2, random_state=rand)
    trainY, validY = train_test_split(trainY, test_size=0.2, random_state=rand)

    # initialize fields that will be used in training
    features = ['bone_length',
                'rotting_flesh',
                'hair_length',
                'has_soul',
                'black', 'blood',
                'blue', 'clear',
                'green', 'white']

    # random guess
    prediction0 = randomGuess(testX[features])
    # k-nearest neighbors
    prediction1 = kNearestNeighbor(trainX[features], trainY, testX[features])
    # 3-layers neural network
    prediction2 = threeLayerNN(trainX[features], trainY, validX[features], validY, testX[features])
    # deep neural network
    prediction3 = deepNN(trainX[features], trainY, validX[features], validY, testX[features])

    # construct output data frame and export to csv
    output(testX, prediction0, "randGuess")
    output(testX, prediction1, "kNearN")
    output(testX, prediction2, "3layerNN")
    output(testX, prediction3, "deepNN")
