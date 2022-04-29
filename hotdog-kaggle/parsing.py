import glob
import numpy as np
from PIL import Image

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
