# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:55:14 2020

@author: gmoha
"""

import numpy as np
from scipy.spatial import distance

def findClosestCentroids(X, centroids):

    (m,n) = X.shape 
    k = centroids.shape[0]
    cent_idx = np.zeros(m, dtype = int)
    
    for i in range(m):
        print("New Data Point")
        d_current = np.zeros((m,1))
        d_old = np.zeros((m,1))
        for j in range(k):
            d_current[i] = distance.euclidean(X[i], centroids[j])
            if j == 0:
                d_old[i] = d_current[i]
                #print("initial")
            else:
                if d_current[i]<d_old[i]:
                    cent_idx[i] = j
                    d_old[i] = d_current[i]
                    #print('min is {0} with {1}'.format(j, d_old[i]))
        print("Mean Centroid distance is {0}".format(np.mean(d_old)))
    return cent_idx
            
                
                
            