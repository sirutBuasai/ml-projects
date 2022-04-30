# external libraries
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
# internal libraries
from parsing import *
from var import *

# Visualization helper
def visualize(img):
    # construct image from flatten image
    img_size = int(img.shape[0]**0.5)
    img = img.reshape((img_size, img_size))
    # visualize the image
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == "__main__":
    # run load_jpg() once on a new machine to download data as .npy
    loadJpg("dog-breed")
    dataX, dataY = load("dog-breed")

    # construct validation set
    rand = np.random.randint(1000)
    trainX, validX = train_test_split(dataX, test_size=0.2, random_state=rand)
    trainY, validY = train_test_split(dataY, test_size=0.2, random_state=rand)

    # initialize neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(120, activation='softmax'))
    model.add(tf.keras.layers.Dense(120, activation='softmax'))
    # set up training and train
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=100)

    # # evaluate on testing data
    loss, acc = model.evaluate(validX, validY)
    print(loss,acc)
