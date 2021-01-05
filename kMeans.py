# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 02:03:38 2020

@author: gmoha
"""

import numpy as np
from findClosestCentroids import findClosestCentroids
from evaluateCentroids import evaluateCentroids
from clusteringLossFunction import KMeansLossFunction


def kMeans(X, centroids, iterations, check_convergence):
    '''
    K-Means Clustering algorithm.
    
    Inputs: 
    Feature Matrix "X", Centroid Coordinates "centroids", # Maximum Iterations "iterations"
    Boolean Value to determine if algorithm terminates at convergence "check_convergence"
    
    Outputs:
    Reevaluated Centroid Coordinates "centroids", training set centroids indices "idx_centroids",
    Number of data points assigned to centroids "cluster_size", loss function value "loss"
    
    '''
    
    for i in range(iterations):
        
        
        k = centroids.shape[0] #get number of wanted clusters k
        idx_centroids = findClosestCentroids(X, centroids) #assign centroids to training data 
        centroids = evaluateCentroids(X, idx_centroids,k) #find new centroids
        
        loss_new = KMeansLossFunction(X, idx_centroids, centroids) #calculate current loss function
        
        if check_convergence:               #Entered if algorithm is to be terminated at convergence
            if i == 0:                      #ensures loss_old initially defined
                loss_old = loss_new
            else:                           #Checks to see if loss function has converged and stops algorithm if so 
                if loss_old <= loss_new:  
                    print("Stopped at iteration {0}.".format(i))
                    break
                loss_old = loss_new
        
        print("Iteration {0} done!".format(i))
        
    cluster_size = centroidCount(centroids, idx_centroids)
        
    return (centroids, idx_centroids, cluster_size, loss_new)


def centroidCount(centroids, idx_centroids):
    
    k = centroids.shape[0]
    a = np.empty(k)
    
    for i in range(k):
        a[i] = np.sum(idx_centroids == i)
        
    return a