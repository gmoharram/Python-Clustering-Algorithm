# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 00:54:02 2020

@author: gmoha
"""

import numpy as np

def evaluateCentroids(X, idx_centroids, k):
    (m,n) = X.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        nearest = X[idx_centroids == i]
        if nearest.size != 0:
            centroids[i] = np.reshape(np.mean(nearest, axis = 0), (1, n))
        
    
    return centroids
    
