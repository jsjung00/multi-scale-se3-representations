from sklearn.neighbors import radius_neighbors_graph
import torch 
import scipy.sparse as sp
import numpy as np 

class SpectralPointCloud:
    '''
    Gets spectral decomposition of our point cloud (encoded as graph). Creates graph from pointcloud,
        calculates eigendecomposition, applies low-pass filter, can get new low pass versions of data matrix
    '''    
    def __init__(self, point_cloud, eps_ball=True, eps_radius=None, k=None):
        '''
        point_cloud: (ND.Array or Torch.tensor)
        eps_ball: (bool) determine if use eps_ball or kNN
        eps_radius: (float)
        k: (int)
        '''
        if torch.is_tensor(point_cloud):
            point_cloud = point_cloud.detach().cpu().numpy()

        self.point_cloud = point_cloud
        self.eps_ball = eps_ball
        self.eps_radius = eps_radius
        self.k = k 
        
        # calculate adjacency matrix
        self.adj_matrix = self.get_adjacency_matrix()

        # get laplacian decomposition 
        self.laplacian_decomposition = self.get_laplacian_decomposition()

    def get_adjacency_matrix(self):
        if self.eps_ball: 
            assert self.eps_radius, "Must provide a radius to define neighbors"
            A = radius_neighbors_graph(self.point_cloud, self.eps_radius, include_self=False)
            assert A[0,0] == 0 and A[1,1] == 0
            # TODO: ensure that diag is zero
            #np.fill_diagonal(A, 0)

        else: 
            assert self.k, "Must provide number of neighbors to define neighbors"
            raise ValueError("K Nearest Neighbors not implemented yet")
        return A 
    
    def get_laplacian_decomposition(self):
        '''
        Returns a list of numpy matrices [U, D, U^T] that represents eigendecomposition of laplacian from   
            adjacency matrix
        '''
        if sp.issparse(self.adj_matrix):
            adj_matrix = self.adj_matrix.toarray()
        else:
            adj_matrix = self.adj_matrix
        
        D = np.diag(np.sum(adj_matrix, axis=1))
        L = D - adj_matrix 

        eigenvalues, eigenvectors = np.linalg.eigh(L)

        eigen_diag = np.diag(eigenvalues) #in ascending order 

        return [eigenvectors, eigen_diag, eigenvectors.T]

    def get_lower_resolution(self, thresh, X):
        '''
        Applies a low pass graph filter on X using the pre-calaculated laplacian decomposition
            thresh: (int) Any eigenvalue index  (in ascending order) larger than thresh will become zero 
        '''
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()

        U, Lambda, U_T = self.laplacian_decomposition

        # Currently do a low pass based on the rank of the eigenvalue, zero out everything larger than thresh
        eigenvalues = np.diag(Lambda).copy()
        eigenvalues[thresh+1:] = 0 # set all indices larger than thresh to zero 

        low_pass_Lambda = np.diag(eigenvalues)    

        thresh_X = U @ low_pass_Lambda @ U_T @ X 
        return thresh_X
    
    def get_lower_resolution_batch(self, len_batch, X, bandwidths=None, lower_band=0.005, upper_band=0.5):
        '''
        Returns: (List[np.ndarr]) a list of len_batch many lower resolution versions of X. 
            If bandwidths not provided, samples uniformly from [lower_band, upper_band] 
        '''
        if bandwidths is None:
            bandwidths = np.random.randint(low=int(lower_band * len(X)), high=int(upper_band * len(X)), size=len_batch)

        thresh_Xs = []
        for band in bandwidths:
            thresh_X = self.get_lower_resolution(band, X)
            thresh_Xs.append(thresh_X)
        
        return thresh_Xs