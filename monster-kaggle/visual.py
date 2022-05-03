# external libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def visualize():
    # read data
    data = pd.read_csv('train.csv')
    # drop id column as we don't need them
    train = data.drop(['id'], axis=1)
    # encode color to numerical data
    le = preprocessing.LabelEncoder()
    le.fit(train['color'])
    train['color_int'] = le.transform(train['color'])

    # PCA plot
    sns.pairplot(train.drop('color', axis=1), hue='type', palette='muted', diag_kind='kde')
    # remove color field for box plot
    train.drop('color_int', axis=1, inplace=True)
    # box plot
    grid = sns.FacetGrid(pd.melt(train, id_vars='type', value_vars=['bone_length',
                                                                 'rotting_flesh',
                                                                 'hair_length',
                                                                 'has_soul']),
                      col='type')
    grid = grid.map(sns.boxplot, 'value', 'variable', palette='muted', order=None)
    # show plot
    plt.show()
