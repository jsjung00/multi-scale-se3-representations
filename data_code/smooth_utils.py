'''
Generate smoother (lower resolution) versions of a point cloud 
'''
import sys
import os 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch 
import scipy.sparse as sp
from scipy.spatial import cKDTree 
import numpy as np 
from collections import deque 
from utils import plot_3d_point_clouds, get_ptcloud_img, get_pointcloud_datasets, get_feat_mask
import pyvista 
import plotly.graph_objects as go
import line_profiler
import time 
import concurrent.futures
import psutil
import pykeops
from pykeops.torch import LazyTensor 


def epanechnikov_torch(d, radius):
    '''
    d: (torch.Tensor)
    radius: (float) bandwidth
    '''
    mu = d / radius 

    breakpoint()
    
    mask = (mu.abs() <= 1).if_then_else(1.0, 0.0)

    return (3.0/4.0) * (1 - mu**2) * mask


def epanechnikov_weight(distances, radius):
    weights = 1 - (distances/radius) ** 2
    weights[distances > radius] = 0
    return 3.0/4.0 * weights

def uniform_weights(distances, radius):
    weights = np.ones_like(distances)
    weights[distances > radius] = 0
    return weights 

def convolve_torch(point_cloud, radius, kernel="epanechnikov"):
    '''
    Returns kernel convolved point cloud and also normalized weight matrix used for convolution

    point_cloud: (torch.Tensor) (N,3)
    radius: (float)

    #TODO: fix and write efficient implementation
    '''
    points_i = LazyTensor(point_cloud[:, None, :]) 
    points_j = LazyTensor(point_cloud[None, :, :])

    D2 = ((points_i - points_j) ** 2).sum(-1).sqrt()

    weights = epanechnikov_torch(D2, radius)
    normalization = weights.sum(dim=1, keepdim=True) + 1e-8
    weights = weights / normalization

    conv_point_cloud = weights @ point_cloud 

    return conv_point_cloud, weights 



def process_radius(radius, point_cloud, feature_matrix, orig_mask, is_tens):
    '''
    Helper function used in create_lowres_input to allow for parallel processing
    '''
    # downsample point cloud 
    smooth_cloud, indices_weights = convolve_point_cloud(point_cloud, radius=radius)

    # downsample feature matrix 
    smooth_features = recreate_convolution(feature_matrix, indices_weights)

    if is_tens:
        smooth_cloud = torch.from_numpy(smooth_cloud).float()
        smooth_features = torch.from_numpy(smooth_features).float()

    # get feats, mask
    feat, mask = get_feat_mask(smooth_cloud, orig_mask)
    return feat, smooth_cloud, mask, smooth_features 


def create_lowres_input(point_cloud, feature_matrix, orig_mask, radius_list):
    '''
    Given original point_cloud and feature matrix, creates downsampled versions of the point cloud and feature matrix
        according to various radii in the radius_list

    point_cloud: (tensor or nd.array) shape (1,N,3) or (N,3)
    feature_matrix: (tensor or nd.array) shape (1,N,d) or (N,d)

    Returns: 
        Tuple of (feats, coors, masks, feature_matrices) where feats is (B,N,1) and coors is (B,N,3) and mask is (B,N) and 
            feature_matrices is (B,N,d)
    '''
    is_tens = torch.is_tensor(point_cloud)

    if is_tens:
        point_cloud = point_cloud.detach().cpu().numpy()
        feature_matrix = feature_matrix.detach().cpu().numpy()
    assert feature_matrix.shape[-1] > 1, "Feature matrix must have a dimension that is larger than 1.... else change squeeze code"

    point_cloud = np.squeeze(point_cloud)
    feature_matrix = np.squeeze(feature_matrix)

    feats, coors, masks, feature_matrices = [], [], [], []

    for radius in radius_list:
        feat, smooth_cloud, mask, smooth_features = process_radius(radius, point_cloud, feature_matrix, orig_mask, is_tens)
        feats.append(feat)
        coors.append(smooth_cloud)
        masks.append(mask)
        feature_matrices.append(smooth_features)

    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = [executor.submit(process_radius, radius, point_cloud, feature_matrix, orig_mask, is_tens) for radius in radius_list]
        for future in concurrent.futures.as_completed(tasks):
            feat, smooth_cloud, mask, smooth_features = future.result()
            feats.append(feat)
            coors.append(smooth_cloud)
            masks.append(mask)
            feature_matrices.append(smooth_features)
    '''
    # torch stack
    feats, coors, masks, feature_matrices = torch.stack(feats), torch.stack(coors), torch.stack(masks), torch.stack(feature_matrices)
    
    return feats, coors, masks, feature_matrices



def recreate_convolution(feature_matrix, indices_weights):
    '''
    Applies the same convolution as dictated by a list of tuples (nbor_indices, weights)

    feature_matrix: (Nd.Array) (N,d)

    Returns:
        smoothed version of the feature matrix, by applying convolution
    '''

    smoothed_features = np.empty_like(feature_matrix)

    for i in range(len(feature_matrix)):
        feature = feature_matrix[i]
        indices, weights = indices_weights[i]

        neighbors = feature_matrix[indices]

        if np.sum(weights) > 0:
            smoothed_features[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0) / np.sum(weights)
        else:
            # no neighbors found
            smoothed_features[i] = feature

    return smoothed_features


@line_profiler.profile 
def convolve_point_cloud(point_cloud, radius, kernel='epanechnikov'):
    '''
    Applies convolution of a given kernel to a point cloud. We make an approximation and only apply local smoothing to neightbors
        within a given radius of a point

    point_cloud: (nd.array) (N,3)  

    Returns: 
        smoothed point cloud (nd.array) (N,3), 
        list of (nbor_indices, weights) tuples to be able to re-create the convolution for a different feature matrix
    '''
    list_nbor_indices_weights = [] # list of (nbor_indices, weights)

    tree = cKDTree(point_cloud)
    smoothed_points = np.empty_like(point_cloud)

    for i in range(len(point_cloud)):
        point = point_cloud[i]
        indices = tree.query_ball_point(point, r=radius)
        
        # convert tuple to indexable?? 
        neighbors = point_cloud[indices]

        distances = np.linalg.norm(neighbors - point, axis=1)

        if kernel == 'epanechnikov':
            weights = epanechnikov_weight(distances, radius)
        elif kernel == 'uniform':
            weights = uniform_weights(distances, radius)
        else:
            raise ValueError("Unknown kernel type")

        if np.sum(weights) > 0:
            smoothed_points[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0) / np.sum(weights)
        else:
            # no neighbors found
            smoothed_points[i] = point
        
        list_nbor_indices_weights.append((indices, weights))

    return smoothed_points, list_nbor_indices_weights


@line_profiler.profile 
def convolve_point_cloud_vectorized(point_cloud, radius, kernel='epanechnikov'):
    """
    Vectorized convolution of a point cloud using pairwise distances.
    
    Parameters:
      point_cloud: np.array of shape (N, 3)
      radius: float, the radius to consider neighbors
      kernel: str, either 'epanechnikov' or 'uniform'
    
    Returns:
      smoothed_points: np.array of shape (N, 3)
      list_nbor_indices_weights: list of tuples (indices, weights) for each point
    """
    N = point_cloud.shape[0]
    
    # Compute pairwise differences and distances: shape (N, N, 3) then (N, N)
    diff = point_cloud[None, :] - point_cloud[:, None]
    dists = np.linalg.norm(diff, axis=2)
    
    # Create a boolean mask for neighbors within the radius
    mask = dists <= radius
    
    # Compute weights vectorized based on kernel type
    if kernel == 'epanechnikov':
        weights = np.clip(1 - (dists / radius) ** 2, 0, None) * mask
    elif kernel == 'uniform':
        weights = mask.astype(np.float64)
    else:
        raise ValueError("Unknown kernel type")
    
    # Sum weights for each point (avoid division by zero)
    sum_weights = weights.sum(axis=1, keepdims=True)
    sum_weights[sum_weights == 0] = 1
    
    # Compute weighted average of neighbors for each point
    smoothed_points = (weights[:, :, None] * point_cloud[None, :]).sum(axis=1) / sum_weights
    
    # Optional: extract neighbor indices and weights for each point.
    # This still requires a Python loop, but only for gathering results, not for heavy computation.
    list_nbor_indices_weights = []
    for i in range(N):
        indices = np.nonzero(mask[i])[0].tolist()
        list_nbor_indices_weights.append((indices, weights[i, mask[i]]))
    
    return smoothed_points, list_nbor_indices_weights



def create_fractal_point_cloud(num_layers=3):
    '''
    Create a fractal point cloud of various cube structures 

    num_layers: number of cube resolutions. num_layers==1 means just a single cube

    Return: (nd.array) (N,3)
    '''
    mult_factor = 0.25
    start_length = 1

    def get_cube_points(lower_left_point, length):
        '''
        Get all points in the cube based on the lower left point in the cube and cube side length

        lower_left_point: (nd.array) (x,y,z) of lower left point of the cube

        Return:
            List[nd.array]
        '''
        points = [] 
        for i in range(0,2):
            for j in range(0,2):
                for k in range(0,2):
                    point = lower_left_point + length * np.array([i,j,k], dtype='f')
                    points.append(point)
        return points 
    
    fractal_points = []
    seed = np.array([0,0,0], dtype='f')

    queue = deque() # list of tuples of (point_arr, unit_length)
    queue.append((seed, start_length))

    while queue:
        top_point, top_length = queue.pop()
        # hack to ensure that seed can also generate a cube around it
        fractal_points.append(top_point)
        
        
        # termination condition  
        if top_length <= mult_factor**num_layers:
            continue 
        
        # get all points in the cube
        all_points = get_cube_points(top_point, top_length)
        for point in all_points:
            queue.append((point,  top_length * mult_factor))
    
    # delete duplicates
    all_points = np.stack(fractal_points)
    return np.unique(all_points, axis=0)


def vista_visualize(point_cloud):
    pdata = pyvista.PolyData(point_cloud)
    sphere = pyvista.Sphere(radius=0.01, phi_resolution=10, theta_resolution=10)
    pc = pdata.glyph(scale=False, geom=sphere, orient=False)
    pc.plot(cmap="Reds")

def plotly_visualize(point_cloud, point_cloud2):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=point_cloud[:,0],
        y=point_cloud[:,1],
        z=point_cloud[:,2],
        mode='markers',
        marker=dict(color='red'),
        name="PC1"
    ))

    fig.add_trace(go.Scatter3d(
        x=point_cloud2[:, 0],
        y=point_cloud2[:, 1],
        z=point_cloud2[:, 2],
        mode='markers',
        marker=dict(color='blue'),
        name='PC2'
    ))

    fig.update_layout(title='3D Point Cloud',
                  scene=dict(
                      xaxis_title='X',
                      yaxis_title='Y',
                      zaxis_title='Z'
                  ))
    fig.show()


if __name__ == "__main__":
    # load some protein structure
    #pc_train_dataset, pc_val_dataset, pc_test_dataset = get_pointcloud_datasets(directory='/mnt/justin/small_data')

    #first_pc = pc_train_dataset[0].cpu().numpy()
    
    test_point_cloud =  create_fractal_point_cloud(num_layers=3)

    convolved_one = convolve_torch(torch.from_numpy(test_point_cloud).float(), radius=0.5)
    convolved_two = convolve_point_cloud(test_point_cloud, radius=0.5)

    checking_convolution = torch.is_equal(convolved_one, torch.from_numpy(convolved_two).float())
    breakpoint()
    
    



    # confirm that the two functions are identical 
    start_time = time.perf_counter()
    convolved_one = convolve_point_cloud(test_point_cloud, radius=0.5)[0]
    end_time = time.perf_counter()
    print(f"Time elapses: {end_time-start_time :.4f} sec" )

    start_time = time.perf_counter()
    convolved_two = convolve_point_cloud_vectorized(test_point_cloud, radius=0.5)[0]
    end_time = time.perf_counter()
    print(f"Time elapses: {end_time-start_time :.4f} sec" )

    breakpoint()

    is_equal = np.all(np.equal(convolved_one, convolved_two))




    #create_lowres_input(torch.from_numpy(test_point_cloud).float(), torch.rand(len(test_point_cloud),100), torch.ones(len(test_point_cloud)), np.random.uniform(size=32))

    #vista_visualize(test_point_cloud)

    # visualize the original point cloud 
    #plotly_visualize(test_point_cloud)

    # visualize the smoothed point cloud

    #smooth_cloud, indices_weights = convolve_point_cloud(test_point_cloud, radius=0.5)
    #plotly_visualize(test_point_cloud, smooth_cloud)

    
    # run some time analysis
    '''
    start_time = time.perf_counter()
    #smooth_cloud, indices_weights = convolve_point_cloud_vectorized(test_point_cloud, radius=0.5)
    smooth_cloud, indices_weights = convolve_point_cloud(test_point_cloud, radius=0.5)
    end_time = time.perf_counter()
    print(f"Time elapses: {end_time-start_time :.4f} sec" )

    start_time = time.perf_counter()
    recreate_smooth_cloud = recreate_convolution(test_point_cloud, indices_weights)
    end_time = time.perf_counter()
    print(f"Time elapsed: {end_time-start_time:.4f} sec")

    is_equal = np.all(np.equal(smooth_cloud, recreate_smooth_cloud))
    print(f"Is the recreation possible: {is_equal}")
    '''
    
    #get_ptcloud_img(test_point_cloud)
    #print(test_point_cloud)
