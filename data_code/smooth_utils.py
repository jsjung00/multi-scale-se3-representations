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
from utils import plot_3d_point_clouds, get_ptcloud_img, get_pointcloud_datasets
import pyvista 
import plotly.graph_objects as go
import line_profiler
import time 


def epanechnikov_weight(distances, radius):
    weights = 1 - (distances/radius) ** 2
    weights[distances > radius] = 0
    return weights

def uniform_weights(distances, radius):
    weights = np.ones_like(distances)
    weights[distances > radius] = 0
    return weights 

def create_lowres_structure_features(point_cloud, feature_matrix, radius_list):
    '''
    Given original point_cloud and feature matrix, creates downsampled versions of the point cloud and feature matrix
        according to various radii in the radius_list

    Returns: 
        List of [(lowres_point_cloud, lowres_feature_matrix)] of original data format
    '''
    is_tens = torch.is_tensor(point_cloud)

    if is_tens:
        point_cloud = point_cloud.detach().cpu().numpy()
        feature_matrix = feature_matrix.detach().cpu().numpy()

    point_cloud_features_pairs = []
    for radius in radius_list:
        # downsample point cloud 
        smooth_cloud, indices_weights = convolve_point_cloud(point_cloud, radius=radius)

        # downsample feature matrix 
        smooth_features = recreate_convolution(feature_matrix, indices_weights)
    
        point_cloud_features_pairs.append((smooth_cloud, smooth_features))
    
    return point_cloud_features_pairs



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




def create_fractal_point_cloud(num_layers=3):
    '''
    Create a fractal point cloud of various cube structures 

    num_layers: number of cube resolutions. num_layers==1 means just a single cube
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

    #vista_visualize(test_point_cloud)

    # visualize the original point cloud 
    #plotly_visualize(test_point_cloud)

    # visualize the smoothed point cloud

    #smooth_cloud, indices_weights = convolve_point_cloud(test_point_cloud, radius=0.5)
    #plotly_visualize(test_point_cloud, smooth_cloud)

    
    # run some time analysis
    start_time = time.perf_counter()
    smooth_cloud, indices_weights = convolve_point_cloud(test_point_cloud, radius=0.5)
    end_time = time.perf_counter()
    print(f"Time elapses: {end_time-start_time :.4f} sec" )

    start_time = time.perf_counter()
    recreate_smooth_cloud = recreate_convolution(test_point_cloud, indices_weights)
    end_time = time.perf_counter()
    print(f"Time elapsed: {end_time-start_time:.4f} sec")

    is_equal = np.all(np.equal(smooth_cloud, recreate_smooth_cloud))
    print(f"Is the recreation possible: {is_equal}")
    
    #get_ptcloud_img(test_point_cloud)
    #print(test_point_cloud)
