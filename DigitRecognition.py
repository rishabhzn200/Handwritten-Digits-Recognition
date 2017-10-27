from binascii import hexlify

import numpy as np
import KNN
import matplotlib.pyplot as plt


"""
Input: 
    datafile: This is the binary file containing normalized image vectors of digits.
    labels: This file contains the labels corresponding to each digits in the datafile in the binary format.
    
Return: 
    List of image data read from the file and the corresponding labels.
"""

def loadData(datafile, labels):

    #read Train Label
    labelfileObj = open(labels, "rb")
    imageLabelData = labelfileObj.read()
    labelfileObj.close()

    #Get the magic number and numLabels
    magicNumberLabel = int(hexlify(imageLabelData[0:4]), 16) #first 4 bytes of label file has magic number.
    numLabels = int(hexlify(imageLabelData[4:8]), 16)    # next four byte of label file has number of labels in the file.

    labelStart = 8  # byte offset in the file where the first label starts

    byteLabel = imageLabelData[labelStart:]
    trainingLabels = np.frombuffer(byteLabel, dtype=np.ubyte)



    #read Train dataset
    datafileObj = open(datafile, "rb")
    imageData = datafileObj.read()
    datafileObj.close()

    #Read magic number and numSamples
    magicNumberData = int(hexlify(imageData[0:4]), 16)  # first 4 bytes of label file has magic number.
    numSamplesInData = int(hexlify(imageData[4:8]), 16)  # next four byte of label file has number of labels in the file.

    #Get the number of rows and coloumns each image represents
    numRows =int(hexlify(imageData[8:12]), 16)
    numCols = int(hexlify(imageData[12:16]), 16)

    dataStartOffset = 16
    imageDataFromB16 = imageData[dataStartOffset:]


    #Convert data to 3d array
    image1DArray = np.frombuffer(imageDataFromB16, dtype=np.ubyte)
    image_2d_array = image1DArray.reshape((numSamplesInData*numRows,numCols))
    image_3d_array = image_2d_array.reshape((numSamplesInData,numRows, numCols))

    #normalize the values
    normalizationFactor = 255.0  #Use 255.0 because in Python 2 255 is treated as integer 255 which is different from Python 3
    image_3d_array_norm = image_3d_array / normalizationFactor

    #return the image Array and the labels
    return [image_3d_array_norm, trainingLabels]


"""
Start of Program:
Read the file and recognize digits using KNN algorithm.
Accuracy is calculated for each value of K.
"""

if __name__ == "__main__":

    #Train and Test data
    X_train, Y_train = loadData("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    X_test, Y_test = loadData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

    X_test = X_test[:5]
    Y_test = Y_test[:5]

    #Implement Knn algorithm
    knnObj = KNN.knn()

    accuracyList = []
    kList = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
    #set the value of k
    for k in kList:
        print("K = " + str(k))

        #Create a list to store the predicted labels for each value of K
        predictedClasses = []
        for index in range(0,X_test.__len__()):

            #find the nearest neighbour of images
            neighbours = knnObj.neighbours(X_train, Y_train, X_test[index], k)

            #find the predicted class of the test image
            predictedClassOfTestImage = knnObj.predictClass(neighbours)

            #Add predicted class of the image to the list. This will be used to find the accuracy later
            predictedClasses.append(predictedClassOfTestImage)

        #find the accuracy of KNN.
        accuracy = knnObj.accuracy(Y_test, predictedClasses)

        #Get the percentage accuracy
        accuracyList.append(accuracy*100)

        print("Accuracy = " + str(accuracy))

    #print("Accuracy List = " + str(accuracyList))

    #plot AccuracyList vs Klist
    plt.plot(kList, accuracyList)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    #plt.show()
    plt.savefig("KNN_Classifier.png")

### End of Program  ###