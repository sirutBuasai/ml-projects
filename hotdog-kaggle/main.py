# external libraries
from matplotlib import pyplot as plt
# internal libraries
from parsing import *
# unused
import tensorflow as tf

# Visualization helper
def visualize(img):
    # construct 48x48 image from flatten image
    img = img.reshape((299,299))
    # visualize the image
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == "__main__":
    # run load_jpg() once on a new machine to download data as .npy
    load_jpg("train")
    load_jpg("test")
    trainX, trainY = load("train")
    testX, testY = load("test")
