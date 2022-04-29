import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
# unused
from sklearn.model_selection import train_test_split
import tensorflow as tf

def compile_ingrdients(files):
    ingredients = []
    for f in files:
        df = pd.read_json(f)
        for recipe in df.ingredients:
            ingredients.extend(recipe)
    return np.array(list(set(ingredients)))

if __name__ == "__main__":
    # load training data set
    data = pd.read_json('train.json')

    # create a new field based on the number of ingredients
    data['num_ingredient'] = data.ingredients.map(lambda a: len(a))
    print(data.head())

    # get total number of classes
    n_classes = len(data['cuisine'].unique())

    # convert each cuisine value to one hot class
    # transformed is a sparse one-hot matrix of cuisine classes in shape (N x C)
    # to convert the sparse matrix to normal matrix, we need transformed.toarray()
    ohe = OneHotEncoder()
    mlb = MultiLabelBinarizer()
    tf_cuisine = ohe.fit_transform(data[['cuisine']])
    tf_ingredients = pd.DataFrame(mlb.fit_transform(data['ingredients']), columns=mlb.classes_, index=data.index)

    print(tf_cuisine.toarray()[0:3])
    print(tf_ingredients.head())

    arr = compile_ingrdients(["train.json", "test.json"])
    print(arr.shape)

