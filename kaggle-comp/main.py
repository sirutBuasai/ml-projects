import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Decode one hot matrix into monster type
def decodeOHE(guess):
    # get the monster type with highest guess percentage
    numerical_value = np.argmax(guess, axis=1)
    # decode the values to monster strings
    decoded = ['Ghost' if n == 0 else 'Ghoul' if n == 1 else 'Goblin' for n in numerical_value]

    return decoded

# Clean the data into a data fram with only numerical fields
def cleanData(data_frame):
    # extract existing numerical data
    data = {'id': data_frame.id,
            'bone_length': data_frame.bone_length,
            'rotting_flesh': data_frame.rotting_flesh,
            'hair_length': data_frame.hair_length,
            'has_soul': data_frame.has_soul}
    # create the cleaned data frame
    cleaned = pd.DataFrame(data)
    # initialize one hot encoder
    # encode color to one hot matrix
    ohe = OneHotEncoder()
    oh_color = ohe.fit_transform(data_frame[['color']])
    # attach the one hot matrix back to data frame
    cleaned[ohe.categories_[0]] = oh_color.toarray()

    return cleaned

# Convert monster category into a categorical value.
# 0 = Ghost, 1 = Ghoul, 2 = Goblin
def categorizeGroundtruth(data_frame):
    # create output data frame
    output = pd.DataFrame()
    # initialize one hot encoder
    # encode type to one hot matrix
    ohe = OneHotEncoder()
    oh_type = ohe.fit_transform(data_frame[['type']])
    # attach the one hot matrix back to data frame
    output[ohe.categories_[0]] = oh_type.toarray()
    
    return output

# Load training data. This function will return both X and Y
def loadTrain(name):
    # read the data set
    df = pd.read_csv(name + ".csv")
    trainX = cleanData(df)
    trainY = categorizeGroundtruth(df)

    return trainX, trainY

# Load testing data. This function will only return X
def loadTest(name):
    df = pd.read_csv(name + ".csv")
    testX = cleanData(df)

    return testX

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

    # cut id field out
    numerical_data = ['bone_length',
                      'rotting_flesh',
                      'hair_length',
                      'has_soul']

    # initialize neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, name="layer1"))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, name="layer2"))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, name="layer3"))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="layer4"))

    # train the neural network
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(trainX[numerical_data], trainY, epochs=20)

    # use weights on testing set
    guess = model.predict(testX[numerical_data])
    predictions = decodeOHE(guess)

    # construct output data frame and export to csv
    output = {'id': testX.id, 'type': predictions}
    res = pd.DataFrame(output)
    res.to_csv("guess.csv", index=False)
