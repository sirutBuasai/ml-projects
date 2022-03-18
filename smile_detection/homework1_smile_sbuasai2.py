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
    yhat = np.array([])
    i = 1
    for image in X:
        i += 1
        # add the result (1,0) onto yhat array
        yhat = np.append(yhat, [predictSmile(predictors, image)])
    # return the fPC of our guesses based on the current set of predictors 
    return fPC(y, yhat)

## Predict whether an image is smiling or not based on the set of predictors
# Argument: Set of tuples of (r1,c1,r2,c2), 2D np.array() of an image
# Return: 1 if the image is a smile, 0 otherwise
def predictSmile (predictors, image):
    prediction = np.array([])
    for p in predictors:
        # declare the pixel pairs (r1,c1), (r2,c2)
        r1,c1,r2,c2 = p[0],p[1],p[2],p[3]
        # if the first pixel is brighter than the second pixel, then we say the prediction_result is 1
        p_res = 1 if image[r1,c1] > image[r2,c2] else 0
        prediction = np.append(prediction, [p_res])
    # return that image is smiling if the predictor set get more
    return 1 if np.mean(prediction) > 0.5 else 0

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

## Same functions as findPredictors but vectorize the coordinates instead of using for loops
# Argument: 3D np.array() of faces, 1D np.array() of labels
# Return: the fPC and the set of predictors
def findPredictors2 (Faces, Labels):
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

## Given a set of predictors from the training set, find the fPC of the predictors on the testing set
# Argument: set of predictor tuples,  3D np.array() of faces, 1D np.array() of labels
def testPredictors (predictors, Faces, Labels):
    guess = np.zeros_like(Labels)
    for i, p in enumerate(predictors):
        r1,c1,r2,c2 = p
        for j in range(len(Faces)):
            guess[j] = 1 if Faces[j,r1,c1] > Faces[j,r2,c2] else 0

    guess = np.where(guess/len(predictors) > 0.5, 1, 0)
    return fPC(Labels, guess)

## Find all possible predictors
# Argument: 3D np.array() of faces, 1D np.array() of labels
# Return: the fPC and the set of predictors
def findPredictors (Faces, Labels):
    """
    - Loop 6 times to get 6 predictors
      - In the first loop:
        - Initiate a current best fPC and current best predictor.
        - Choose a pair of pixels to compare {r1,c1}.
        - Loop through all the training data to and get the fPC of the predictor.
        - Store the fPC and the predictor if it is the current best.
        - After the loop, get the current best predictor.
      - In the following loops:
        - Initiate a current best fPC and current best predictor.
        - Choose a pair of pixels to compare {r1,c1,r2,c2}.
        - Loop through all the training data and get the fPC of the predictor ALONG WITH THE PREVIOUSLY CHOSEN PREDICTOR.
        - Store the fPC and the predictor if it is the current best.
        - After the loop, get the current best predictor.
    """
    final_phi = set()
    for i in range(1):
        max_fPC = 0
        max_phi = (0,0,0,0)
        for r1,c1 in np.ndindex((24,24)):
            for r2,c2 in np.ndindex((24,24)):
                curr_phi = (r1,c1,r2,c2)
                if (r1 == r2 and c1 == c2) or curr_phi in final_phi:
                    continue
                else:
                    curr_fPC =  measureAccuracyOfPredictors(final_phi | {(r1,c1,r2,c2)}, trainingFaces[:10], trainingLabels[:10])
                    if curr_fPC > max_fPC:
                        max_fPC = curr_fPC
                        max_phi = curr_phi

        final_phi.add(max_phi)

    return max_fPC, final_phi

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    print(fPC(np.array([1,1,1,1,1]),np.array([1,1,1,2,2])))
    print(len(testingFaces))
    print(trainingLabels)
    print(np.mean(np.array([1,1,0])))
    print(predictSmile({(0,0,0,1)},trainingFaces[0]))
    print(measureAccuracyOfPredictors({(0,0,0,1)}, trainingFaces, trainingLabels))
    # res, pred = findPredictors(trainingFaces, trainingLabels)
    # res, pred = findPredictors2(trainingFaces, trainingLabels)
    # print(res, pred)
    # stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, pred)
    pred = {(19, 11, 14, 7), (12, 5, 10, 13), (11, 19, 12, 12), (14, 5, 16, 6), (20, 7, 17, 7), (20, 17, 16, 17)}
    print(testPredictors(pred, testingFaces, testingLabels))
