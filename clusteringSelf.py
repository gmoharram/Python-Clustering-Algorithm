import numpy as np
import pandas as pd

from NormalizeMatrixColumns import normalizeColumns 

from initializeCentroids import initializeCentroids  
from kMeans import kMeans 





#Import data and remove invoice number info
df = pd.read_excel('File.xlsx')
#df.drop(labels ='Unnamed: 0', axis=1, inplace = True) #In case dataframe generates index column

#Create numpy feature matrix
X = df.to_numpy()

#normalize feature matrix columns
(X, X_norm) = normalizeColumns(X) 



#Generate initial k centroids by using random sample of X
k = 5
centroids = initializeCentroids(X, k)

#Set Number of maximum iterations to perform
iterations = 500

#Determine if algorithm should be terminated if convergence of loss function is assumed to have been reached
check_convergence = True

#Run k-means algorithm 
(centroids, idx_centroids, cluster_size, loss_function_value) = kMeans(X, centroids, iterations, check_convergence)

#Undo normalization
real_centroids = centroids * X_norm






 

