import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # load training data set
    data_tr = pd.read_json('train.json')

    # create a field for the amount of ingredients
    data_tr['length'] = data_tr.ingredients.map(lambda a: len(a))

    # create a field for labeling cuisine categorically
    le = LabelEncoder()
    data_tr['categorical_label'] = le.fit_transform(data_tr.cuisine)

    # split the training data into training and validation set
    # stratified on cuisine category, random_state for reproduction
    trainX, validX = train_test_split(data_tr, test_size=0.15, stratify=data_tr.cuisine, random_state=42)

    # get labels of the training and validation set
    trainY, validY = trainX.categorical_label, validX.categorical_label

    # get the ingredient list for each of the dish
    train_ingredients = [','.join(ingredient) for ingredient in trainX.ingredients.values.tolist()]
    valid_ingredients = [','.join(ingredient) for ingredient in validX.ingredients.values.tolist()]

