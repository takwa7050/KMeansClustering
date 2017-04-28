"""
Author: Tousif Chowdhury
Date: 04/10/2017
Description: An algorithm to run the K-Means Clustering Algorithm.

Quick Overview of the Alforithm:
Input K, data points (x1...xn)
Place Centroids at random locations (c1....cj)
Repeat until convergance:
    For each point x:
        Find the nearest centroid C(j), using Euclidean Distance
        Assign the point xi to the cluster j
    For each cluster j = 1 though k
        Calculate the new centroid, cj= mean of all points assigned in previous step

Stop When none of the clusters assignments don't change
"""
import random
import pandas as pd
import math
import matplotlib.pyplot as plt

class Cluster(object):
    """
    Object to represent what will be in the cluster
    """

    """
    Points- A list of lists to hold the cooridantes in the cluster
    Centroid - A coordinate to represent the centroid (Need to loop through list and find closest point)
    """
    def __init__(self,points):
        """
        """
        self.points = points
        self.dimension = len(points[0])
        self.centroid = self.calculateCentroid()

    def getXValues(self):
        xValues = []
        for points in self.points:
            x = points[0]
            xValues.append(x)

        return xValues

    def getYValues(self):
        YValues = []
        for points in self.points:
            y = points[1]
            YValues.append(y)

        return YValues

    def calculateCentroid(self):
        """
        Find the center for the points in the list of coordiantes
        """
        if (len(self.points) != 0):
            numberOfPoints = len(self.points)
            coordinates = []
            for points in self.points:


                coordinates.append(points)

                unZippedPoints = zip(*coordinates)
                centroid = [math.fsum(point)/numberOfPoints for point in unZippedPoints]

            return centroid
        else:
            return [0,0,0]


    def update(self, points):
        """
           Function to calculate how much the cluster moved
           """
        oldCentroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        # print("The old centroid is "+ str(oldCentroid))
        # print("The new centroid is " + str(self.centroid))
        moved = calculateDistance(oldCentroid, self.centroid)

        return  moved

def sse(clusterArray):
    """
    error : Distance from each point to the centroid squared
    Add to the sum all the distances
    :param clusterArray: Take in an array of clusters
    :return: the sse of the cluster
    """
    sumCounter = 0
    for cluster in clusterArray:
        for point in cluster.points:
            distance = calculateDistance(point,cluster.centroid)
            sqauredDist = distance**2
            sumCounter = sumCounter + sqauredDist

    return sumCounter

def kmeans(points,dimensions,stoppingCondition):

    initialValues = random.sample(points, dimensions)

    clusters = [Cluster([points]) for points in initialValues]

    counter = 0
    while True:
        listOfPointsinCluster = [[] for _ in clusters]
        clusterCount = len(clusters)
        counter = counter + 1
        for point in points:
            smallestDist = calculateDistance(point, clusters[0].centroid)
            index = 0

            for i in range(clusterCount-1):
                distance = calculateDistance(point,clusters[i+1].centroid)

                if distance < smallestDist:
                    smallestDist = distance
                    index = i+1
            listOfPointsinCluster[index].append(point)

        jump = 0.0

        for i in range(clusterCount):
            moved = clusters[i].update(listOfPointsinCluster[i])

            jump = max(jump,moved)

        if jump < stoppingCondition:
            break
    return clusters

"""
Function to run kmeans at least 10 times in this to get the best random points and SSE
"""
def runKmeans(dataSets, k, stoppingCriteria):
    listOfClusters = []

    counterTest = 0
    while counterTest <= 10:
        cluster = kmeans(dataSets,k,stoppingCriteria)
        listOfClusters.append(cluster)
        counterTest = counterTest + 1

    smallestSSE = 100000
    counter = 0
    for i in range(0,len(listOfClusters)):
        SSE = sse(listOfClusters[i])

        if SSE < smallestSSE:
            smallestSSE = SSE
            counter = i

    return listOfClusters[counter]
"""
Function to calculate the euclidian distance between two points
"""
def calculateDistance(list1, list2):
    counter = 0
    for i in range(0,len(list1)):
        value = list1[i] - list2[i]
        valueSquare = math.pow(value,2)
        counter = counter + valueSquare

    euclid = math.sqrt(counter)

    return euclid


def main():
    """
    Read in the data from the csv
    """
    df = pd.read_csv('testdata.csv', delimiter=',')
    listOfData = [list(x) for x in df.values]

    """
    Record each sse for k = 1 through 20
    """
    sseArray = []
    for i in range(1, 21):
        clusters = runKmeans(listOfData,i,0.2)
        sseVal = sse(clusters)
        sseArray.append(sseVal)

    kValues = []
    for i in range(1,21):
        kValues.append(i)

    """
    Plot the SSEs to find the knee or inflection point, in this case its 5
    """
    plt.figure(1)
    plt.ylabel("SSE")
    plt.xlabel("K")
    plt.title("Best K Value")
    plt.plot(kValues, sseArray, "-ro", markersize=3)
    plt.show()

    """
    Run k means with k = 5 and get the clusters and display all the values.
    """
    actualClusters = kmeans(listOfData,5,0.2)
    sseVal = sse(actualClusters)
    print("----------------------------------------------------------------------------------------")
    print("The SSE is " + str(sseVal))
    for cluster in actualClusters:
        print("----------------------------------------------------------------------------------------")
        x = cluster.points
        print("The number of points in this cluster " + str(len(x)))
        print("The centroid is: "+ str(cluster.centroid))
        print("----------------------------------------------------------------------------------------")

    colors = ['bo','ro','go','yo','ko']
    plt.figure(2)
    plt.ylabel("Y Values")
    plt.xlabel("X Values")
    plt.axis([0, 10, 0, 10])
    plt.title("Clusters")
    for i in range(0,len(actualClusters)):
        testCluster = actualClusters[i]
        plt.plot(testCluster.getXValues(), testCluster.getYValues(), colors[i])
    plt.show()

if __name__ == '__main__':
    main()




