import numpy as np

def initializeCentroids(X,k):
    (m,n) = X.shape
    if k <= m:
        idx = np.random.randint(0, high = m - 1, size =k  )
        centroids = X[idx]
    else:
        print("Reduce Number of centroids to be less than the number of data points!")
        return
    return centroids