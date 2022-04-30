# external libraries
import glob
import copy
import numpy as np
import tensorflow as tf
from PIL import Image
# internal libraries
from var import *

# Helper to get the image size statistics for ease of resizing images
def getImgStats(path):
    heights, widths = [], []
    for sub_class in glob.glob(path):
        for img_file in glob.glob(sub_class + "*.jpg"):
            # open image with pillow and grayscale it
            img = Image.open(img_file).convert('L')
            # convert image file to numpy array
            np_img = np.array(img, dtype=np.uint8)
            heights.append(np_img.shape[0])
            widths.append(np_img.shape[1])
    avg_h = np.mean(heights)
    avg_w = np.mean(widths)
    return avg_h, avg_w

# Helper for reading an image and convert it to numpy
def convert(img_list, img_labels, path, img_size=IMG_SIZE):
    n_class = 0
    num_class = len(glob.glob(path))
    print("Reading .jpg files and converting them to numpy...")
    for sub_class in glob.glob(path):
        for img_file in glob.glob(sub_class + "*.jpg"):
            # open image with pillow and grayscale it
            img = Image.open(img_file).convert('L')
            # resize images to a standardized size
            img = img.resize((img_size, img_size), Image.ANTIALIAS)
            # convert image file to numpy array
            np_img = np.array(img, dtype=np.uint8).flatten()
            # data augmentation to double the amount of data
            # np_flip = np.fliplr(np.array(img, dtype=np.uint8)).flatten()
            # append the image to over all array
            img_list.append(np_img)
            # img_list.append(np_flip)
            # construct one-hot classification
            label = [0 for _ in range(num_class)]
            label[n_class] = 1
            # flip_label = copy.deepcopy(label)
            # add label twice to account for the flipped image
            img_labels.append(label)
            # img_labels.append(flip_label)
        # increment class
        n_class+=1
    return np.array(img_list), np.array(img_labels)

# Helper to convert jpg to npy and save them
def loadJpg(name):
    # load data set
    x_list, y_list = convert([], [], name + "/*/")
    # shuffle the data to mix hotdog and nothotdog
    print("Permuting data...")
    rand_idx = np.random.permutation(x_list.shape[0])
    x_list = x_list[rand_idx, :]
    y_list = y_list[rand_idx, :]
    # download as .npy for ease of use
    print("Normalizing data...")
    x_list = tf.keras.utils.normalize(x_list, axis=1)
    print("Saving images as .npy...")
    np.save(name + "X.npy", x_list)
    np.save(name + "Y.npy", y_list)

# Helper to load npy into numpy array
def load(name):
    x_list = np.load(name + "X.npy")
    y_list = np.load(name + "Y.npy")
    return x_list, y_list
