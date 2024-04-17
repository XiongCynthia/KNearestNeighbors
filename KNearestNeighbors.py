import torch

class KNearestNeighbors:
    '''
    Class for calculating k-nearest neighbors in a dataset.
    '''
    def __init__(self, X:torch.Tensor):
        '''
        Args:
            X: Training data
        '''
        self.X = X
        return


    def get_nearest_neighbors(self, X:torch.Tensor, n_neighbors=5, return_idx=False):
        '''
        Use training data to calculate k-nearest neighbors for each sample observation.

        Args:
            X: Sample data
            n_neighbors: The number of neighbors to count for
            return_idx: Return the indices of the nearest neighbors in X if False; otherwise return the original values
        '''        
        # Run through each observation in the sample data
        nearest_neighbors = torch.zeros(size=(len(X), n_neighbors)) # Store neighbors nearest to each sample observation
        for sample_i, sample_obs in enumerate(X):
            # Compute distances between the current sample observation and every observation in the training data
            dists = [0] * len(self.X)
            for train_i, train_obs in enumerate(self.X):
                 dists[train_i] = self.__dist(sample_obs, train_obs)
            # Get indices of nearest neighbors
            neighbors = torch.argsort(torch.tensor(dists))[:n_neighbors]
            nearest_neighbors[sample_i] = neighbors
        
        if return_idx:
            return nearest_neighbors
        else:
            nearest_neighbors = nearest_neighbors.int()
            return self.X[nearest_neighbors]
        
                
    def __dist(self, u, v):
        '''Computes Euclidean distance'''
        return torch.sqrt(torch.sum( torch.tensor([(u[i]-v[i])**2 for i in range(v.shape[0])]) ))
    