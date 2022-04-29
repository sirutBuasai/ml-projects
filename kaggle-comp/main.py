# external libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

def cleanData(data_frame):
    # extract existing numerical data
    numerical_data = {'bone_length': data_frame.bone_length,
                      'rotting_flesh': data_frame.rotting_flesh,
                      'hair_length': data_frame.hair_length,
                      'has_soul': data_frame.has_soul}
    # create the cleaned data frame
    cleaned = pd.DataFrame(numerical_data)
    # initialize one hot encoder
    # encode color to one hot matrix
    ohe = OneHotEncoder()
    oh_color = ohe.fit_transform(data_frame[['color']])
    # attach the one hot matrix back to data frame
    cleaned[ohe.categories_[0]] = oh_color.toarray()

    return cleaned

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

def loadTrain(name):
    # read the data set
    df = pd.read_csv(name + ".csv")
    trainX = cleanData(df)
    trainY = categorizeGroundtruth(df)
    print("yes")
    return trainX, trainY

def loadTest(name):
    df = pd.read_csv(name + ".csv")
    testX = cleanData(df)
    return testX

if __name__ == "__main__":
    # read data
    trainX, trainY = loadTrain("train")
    testX = loadTest("test")

    # initialize neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, name="layer1"))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, name="layer2"))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="layer3"))

    # set up training and train
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=20)

    # use weights on testing set
    predictions = model.predict(testX)
    res = np.argmax(predictions, axis=1)
    print(predictions[:5])
    print(res[:5])
    name = []
    for n in res:
        if n == 0:
            name.append('Ghost')
        elif n == 1:
            name.append('Ghoul')
        else:
            name.append('Goblin')

    res_data = {'id': df_te.id,
                'type': name}
    res = pd.DataFrame(res_data)
    res.to_csv("out.csv", index=False)
