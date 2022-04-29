import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# unused
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import json

if __name__ == "__main__":
    # load training data set
    data = pd.read_json('train.json')

    # create a new field based on the number of ingredients
    data['length'] = data.ingredients.map(lambda a: len(a))

    # get total number of classes
    n_classes = len(data['cuisine'].unique())

    # convert each cuisine value to one hot class
    # transformed is a sparse one-hot matrix of cuisine classes in shape (N x C)
    # to convert the sparse matrix to normal matrix, we need transformed.toarray()
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(data[['cuisine']])

    print(data.ingredients.to_numpy()[0:2])
    print(len(data.ingredients.tolist()))

    ################################################################################
    # df = pd.read_json('train.json')
    # n_classes = len(df['cuisine'].unique())
    # print("Number of classes", n_classes)

    # # get the length of the tokens
    # df['length'] = df.ingredients.map(lambda x: len(x))

    # # get the number of classes
    # le = LabelEncoder()
    # df['categorical_label'] = le.fit_transform(df.cuisine)

    # # split dataset
    # train_set, valid_set = train_test_split(df, test_size=0.15, stratify=df.cuisine, random_state=42)

    # train_sentences = [','.join(sentence) for sentence in train_set.ingredients.values.tolist()]
    # valid_sentences = [','.join(sentence) for sentence in valid_set.ingredients.values.tolist()]

    # # get the labels
    # y_train = train_set.categorical_label
    # y_valid = valid_set.categorical_label


    # # get sequence max length
    # sequence_length = int(df['length'].max())

    # # create vectorization layer
    # vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=None, output_mode='int', output_sequence_length=sequence_length, 
    #                                                         split=lambda x: tf.strings.split(x, ','), standardize=lambda x: tf.strings.lower(x))
    # vectorization_layer.adapt(train_sentences)

    # # create vectorization layer
    # vectorizer = tf.keras.models.Sequential()
    # vectorizer.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    # vectorizer.add(vectorization_layer)

    # # get sequences
    # train_sequences = vectorizer.predict(train_sentences)
    # valid_sequences = vectorizer.predict(valid_sentences)

    # print(len(vectorization_layer.get_vocabulary()))
    # print(vectorization_layer.get_vocabulary()[:10])
