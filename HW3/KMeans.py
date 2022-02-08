import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

class KMeans():

    def ArbitraryCentroid(self):
        centroid = np.random.choice(self.X.shape[0], 1) 
        self.Centroid[0] = self.X[centroid]
            
        for i in range(1, self.K):
            distance =  euclidean_distances(self.X, self.Centroid[0:i])
            self.Centroid[i] = self.X[np.argmax(np.amin(distance, axis = 1))]
                
    def DistanceAndCluster(self):
        self.Distance = euclidean_distances(self.X, self.Centroid)
        self.Y = np.argmin(self.Distance, axis = 1)
        
        
    def Centroids(self):  
        listOfCentroids = np.zeros((self.K, self.Feature))
        
        for i in range(self.K):
            listOfCentroids[i] = self.X[self.Y == i].mean(axis = 0)
            
        self.Centroid = np.asarray(listOfCentroids)                    

        
    def Errors(self):
        listOfErrors = np.zeros((self.K))
        
        for i in range(self.K):
            listOfErrors[i] = (np.min(self.Distance[self.Y == i], axis = 1)).sum()
            
        self.Error = np.asarray(listOfErrors)
        
    def Main(self, iteration):
        for i in tqdm(range(iteration)):
            self.DistanceAndCluster()
            self.Centroids()
            self.Errors()            
        
        return self.Error, self.Y, self.Centroid
    
    def __init__(self, X, K, F):
        self.K = K
        self.X = X
        self.Y = None
        self.Distance = np.zeros((X.shape[0], K))
        self.Error = None
        self.Centroid = np.zeros((K, X.shape[1]))
        self.Feature = F