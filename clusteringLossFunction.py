# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 22:14:43 2021

@author: gmoha
"""

from scipy.spatial import distance

def KMeansLossFunction(X,labels, centroids):
    
    (m,n)  = X.shape
    loss = 0
    for i in range(m):
        loss += distance.euclidean(X[i], centroids[labels[i]])
    
    return loss
    