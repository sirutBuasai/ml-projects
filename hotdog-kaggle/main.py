import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
# unused
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

def convertAndLabel(img_list, img_labels, set_name, class_name):
    for img_file in glob.glob(set_name + "/" + class_name + "/*.jpg"):
        # open image with pillow and grayscale it
        img = Image.open(img_file).convert('L')
        # convert image file to numpy array
        np_img = np.array(img.getdata(), dtype=np.uint8)
        # append the image to over all array
        img_list.append(np_img)
        # append the corresponding labels
        if class_name == "hotdog":
            img_labels.append([1,0])
        elif class_name == "nothotdog":
            img_labels.append([0,1])
    return np.array(img_list), np.array(img_labels)

def load_jpg(name):
    # load data set
    hd_list, hd_labels = convertAndLabel([], [], name, "hotdog")
    nhd_list, nhd_labels = convertAndLabel([], [], name, "nothotdog")
    # combind hotdog and nothotdog data together
    x_list = np.append(hd_list, nhd_list, axis=0)
    y_list = np.append(hd_labels, nhd_labels, axis=0)
    # shuffle the data to mix hotdog and nothotdog
    rand_idx = np.random.permutation(x_list.shape[0])
    x_list = x_list[rand_idx, :]
    y_list = y_list[rand_idx, :]
    # download as .npy for ease of use
    np.save(name + "X.npy", x_list)
    np.save(name + "Y.npy", y_list)

def load(name):
    x_list = np.load(name + "X.npy")
    y_list = np.load(name + "Y.npy")
    return x_list, y_list

# Visualization helper
def visualize(img):
    # construct 48x48 image from flatten image
    img = img.reshape((299,299))
    # visualize the image
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == "__main__":
    load_jpg("train")
    load_jpg("test")
    trainX, trainY = load("train")
    testX, testY = load("test")
    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)
    visualize(testX[0])
    visualize(testX[-1])
    print(testY[0], testY[-1])

