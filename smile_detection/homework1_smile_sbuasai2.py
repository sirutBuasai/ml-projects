import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

## Percent Correct
# Argument: 1D np.array() of ground truth, 1D np.array() of guesses
# Return: the percent correct between guesses (yhat) and ground truth (y)
def fPC (y, yhat):
    return np.mean(y == yhat)

## Measure the accuracy of the given set of predictors
# Argument: Set of tuples of (r1,c1,r2,c2), 3D np.array() of images, the np.array() of ground truth
# Return: the percent correct given the set of predictors
def measureAccuracyOfPredictors (predictors, X, y):
    yhat = np.zeros_like(y)
    for i, p in enumerate(predictors):
        r1,c1,r2,c2 = p
        for j in range(len(X)):
            yhat[j] += 1 if X[j,r1,c1] > X[j,r2,c2] else 0

    yhat = np.where(yhat/len(predictors) > 0.5, 1, 0)
    return fPC(y, yhat)

## step-wise regression building on 6 rounds of predictors
# Argument: 3D np.array() of training faces, 1D np.array() of training labels, 3D np.array() of testing faces, 1D np.array() of testing labels
# Return:
def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels, predictors):
    show = True
    for i in range(10):
        # Show an arbitrary test image in gray scale
        im = trainingFaces[i,:,:]
        fig, ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        for p in predictors:
            r1,c1,r2,c2 = p
            if show:
                # Show r1,c1
                rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # Show r2,c2
                rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
                # Display the merged result
        plt.show()

## load .npy data into an np.array()
# Argument: the file name
# Return: 3D np.array() of faces and 1D np.array() of labels
def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

## Find 6 pairs of predictors that best predict the training images
# Argument: 3D np.array() of faces, 1D np.array() of labels
# Return: the fPC and the set of predictors
def findPredictors (Faces, Labels):
    # initialize the final sets of predictors and the preditions from previous set or preditors
    final_phi = set()
    prev_guess = np.zeros_like(Labels)
    for i in range(6):
        print(f"training round {i}!")
        # initialize best fPC, best predictor tuple, and best guess in the current round
        max_fPC = 0
        max_phi = (0,0,0,0)
        max_diff = np.zeros_like(Labels)
        # choose r1,c1 as a base partial feature
        for r1,c1 in np.ndindex(Faces[0].shape):
            # for each pixel (r1,c1) fidn the diff over ALL the images
            diff = np.zeros_like(Faces)
            for j, face in enumerate(Faces):
                # if the pixel (r1,c1) is higher than the target then change the diff value to 1, 0 otherwise
                diff[j] = np.where((face - face[r1,c1]) < 0, 1, 0)
            # choose the second pixel (r2,c2) that has been compared with (r1,c1)
            for r2,c2 in np.ndindex(diff[0].shape):
                curr_phi = (r1,c1,r2,c2)
                # if first and second pixel are not the same and the predictor has not been used yet
                if not ((r1 == r2 and c1 == c2) or curr_phi in final_phi):
                    # find the fPC of the current guess w.r.t to ground truth
                    # current guess is the previous guess from previous predictors + guess from (r1,c1,r2,c1)
                    curr_guess = np.where((prev_guess + diff[:,r2,c2])/(i+1) > 0.5, 1, 0)
                    curr_fPC = fPC(Labels, curr_guess)
                    # if the current predictor (r1,c1,r2,c2) is the best seen so far, change max_fPC, max_phi, max_diff 
                    if curr_fPC > max_fPC:
                        max_fPC = curr_fPC
                        max_phi = (r1,c1,r2,c2)
                        max_diff = diff[:,r2,c2]

        prev_guess = prev_guess + max_diff
        final_phi.add(max_phi)

    return max_fPC, final_phi

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    res, pred = findPredictors(trainingFaces, trainingLabels)
    print(res, pred)
    print(measureAccuracyOfPredictors(pred, testingFaces, testingLabels))
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, pred)
