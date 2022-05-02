# external libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Decode one hot matrix into monster type
def decodeOHE(guess):
    # get the monster type with highest guess percentage
    numerical_value = np.argmax(guess, axis=1)
    # decode the values to monster strings
    decoded = ['Ghost' if n == 0 else 'Ghoul' if n == 1 else 'Goblin' for n in numerical_value]

    return decoded

# Clean the data into a data frame with only numerical fields
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
# idx 0 = Ghost, 1 = Ghoul, 2 = Goblin
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

# Take a result prediction and output as csv
def output(testX, prediction, file_name):
    output = {'id': testX.id, 'type': prediction}
    result = pd.DataFrame(output)
    result.to_csv(file_name + ".csv", index=False)
