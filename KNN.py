
import numpy as np
import math
import operator

class knn:

    def __init__(self):
        pass


    """
    Input: 
        image1: This is the matrix of 28*28
        image2: This is the matrix of 28*28
        
    Return:
        distance: This is the sum of euclidean distance between each point in image1 and image2
        
    This function calculates and returns the euclidean distance between the two images.
    """
    def euclideanDistance(self, image1, image2):

        distance = np.sqrt(np.sum(np.square(image2 - image1)))

        return distance

        pass


    """
    Input:
        trainingData: These are the 60000 samples of 28*28 images.
        labels: These are the labels corresponding to the trainingData.
        testExample: This is the 28*28 image whose nearest neighbours are to be found
        k: number of nearest neighbours to be returned
        
    Output:
        kNearestNeighbours: List of k nearest neighbours of testExample. 
                            Each element in the list has nearestImage and the corresponding label
                            
    This function finds and returns the k nearest neighbours of the input testExample. 
        
    """
    def neighbours(self, trainingData, labels,  testExample, k):

        neighbours = []

        for indexTrainImage in range(0, trainingData.__len__()):
            distance = self.euclideanDistance(trainingData[indexTrainImage], testExample)

            #print(str(indexTrainImage))
            neighbours.append([trainingData[indexTrainImage], distance, labels[indexTrainImage]])

        neighbours.sort(key=operator.itemgetter(1))

        kNearestNeighbours = []

        for i in range(0,k):
            kNearestNeighbours.append( [neighbours[i][0], neighbours[i][2]])  #[i][0] is we are returning first image and not the distance

        return kNearestNeighbours


    """
    Input:
        nearestNeighbours: list of nearest neighbours and corresponding labels fo testExample
        
    Output:
        classPredictionDict_sorted_keys[0]: the class label
        
    This function takes the nearest neighbours of the testExample and finds out and returns the class of majority of neighbours
    """

    def predictClass(self, nearestNeighbours):
        classPredictionDict = {}

        for neighbour in nearestNeighbours:
            if neighbour[1] in classPredictionDict.keys():
                classPredictionDict[neighbour[1]] += 1
            else:
                classPredictionDict[neighbour[1]] = 1

        classPredictionDict_sorted_keys = sorted(classPredictionDict, key=classPredictionDict.get, reverse=True)
        return classPredictionDict_sorted_keys[0]


    """
    Input:
        testLabel: This is the list of test labels
        predictedLabel: This is the list of labels predicted by KNN
        
    Output:
        accuracy_score: This is the accuracy of the knn in the range of 0 and 1
        
    This function finds and returns the accuracy of the KNN algorithm.
    """

    def accuracy(self, testLabel, predictedLabel):

        correctPredictions = 0

        for i in range(0,testLabel.__len__()):

            if testLabel[i] == predictedLabel[i]:
                correctPredictions += 1

        accuracy_score = float(correctPredictions) / testLabel.__len__()

        return accuracy_score