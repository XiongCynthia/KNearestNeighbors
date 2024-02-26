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


    def get_nearest_neighbors(self, X:torch.Tensor, n_neighbors=5):
        '''
        Use training data to calculate k-nearest neighbors for each sample observation.

        Args:
            X: Sample data
            n_neighbors: The number of neighbors to count for
        '''        
        # Run through each observation in the sample data
        nearest_neighbors = [] # Store neighbors nearest to each sample observation
        for sample_obs in X:
            # Compute distances between the current sample observation and every observation in the training data
            neighbors = [] # Store neighbors nearest to the sample observation
            dists = [] # Store distances of the nearest neighbors
            for train_obs in self.X:
                dist = self.__dist(sample_obs, train_obs)
                if len(dists) < n_neighbors:
                    neighbors.append(train_obs)
                    dists.append(dist)
                elif dist < max(dists):
                    # Replace furthest training observation
                    idx = dists.index(max(dists))
                    neighbors[idx] = train_obs.tolist()
                    dists[idx] = dist
            nearest_neighbors.append(neighbors)
        
        return torch.Tensor(nearest_neighbors)

                
    def __dist(self, u, v):
        '''Computes Euclidean distance'''
        return torch.sqrt(torch.sum( torch.Tensor([(u[i]-v[i])**2 for i in range(v.shape[0])]) ))
    