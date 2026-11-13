import numpy as np
import constants # Imports the custom constants module which contains values like mean and standard deviation
from fancy_pca import FancyPCA # Imports the FancyPCA class from fancy_pca.py
# Imports gaussian_filter from scipy to perform Gaussian smoothing -> g(x) = 1/(sqrt(2*pi)*sigma) * exp(-x^2/(2*sigma^2))
from scipy.ndimage import gaussian_filter


# Changes colors based on PCA and smooths the clusters

# From Fancy PCA paper
def _fancy_pca_vectors(data, cluster_count):
    standard_deviation = constants.FANCY_PCA_STANDARD_DEVIATION
    mean = constants.FANCY_PCA_MEAN
    mean_free_data = data - np.mean(data) # Subtract the mean from data (center the data)
    data_covarianz = np.cov(mean_free_data, rowvar=False) # Compute covariance matrix, rowvar=False -> variables in columns
    eig_values, eig_vecs = np.linalg.eigh(data_covarianz) # Compute shared eigenvalues and eigenvectors once for the entire image
    alphas = np.random.randn(cluster_count, 3) * standard_deviation + mean # Generate three independent alpha coefficients for each cluster
    eig_values_list = eig_values * alphas # Scale the shared eigenvalues separately for each cluster
    final_vecs = (eig_vecs @ eig_values_list.T).T # One RGB vector per cluster: a_k = sum_i alpha[k,i] * lambda[i] * p[:,i]
    return final_vecs # Return the calculated vectors, shape (cluster_count, 3)


def modify_clusters(data, pixel_cluster_map, cluster_count, size_images, data_index):
    data_modify = data.copy() # Copy input data to modify without changing original
    add_vecs = _fancy_pca_vectors(data_modify, cluster_count) # Compute one shared image PCA and a separate alpha vector for each cluster
    add_vecs_smooth = _smooth_add_vecs(pixel_cluster_map, size_images, add_vecs, data_index) # Smooth vectors based on cluster info if enabled
    data_modify += add_vecs_smooth # Add smoothed vectors to original data
    clipped_data = np.clip(data_modify, 0, 1) # Clip data to range [0, 1]
    return clipped_data # Return modified data


def _smooth_add_vecs(pixel_cluster_map, size_images, add_vecs, data_index):
    vector_field = list() # Initialize list for vector field data (vectors for each pixel)
    for cluster_idx in pixel_cluster_map: # Loop over all cluster indices in cluster mapping
        vector_field.append(add_vecs[int(cluster_idx)]) # Append vector for current cluster
    vector_field = np.array(vector_field) # Convert list of vectors to numpy array

    # Reshape vector field to match image dimensions, height = size_images[data_index][1], width = size_images[data_index][0], 3 = RGB channels
    vector_field = vector_field.reshape((size_images[data_index][1], size_images[data_index][0], 3))

    if constants.USE_SMOOTH: # Check if smoothing is enabled (True)
        smoothed_vector_field = np.zeros_like(vector_field) # Create an array with same shape as vector_field filled with zeros

        for i in range(3): # Loop over the 3 color channels (RGB)
            # Smooth each color channel using a Gaussian filter, sigma=constants.SIGMA -> standard deviation of Gaussian filter
            smoothed_vector_field[:, :, i] = gaussian_filter(vector_field[:, :, i], sigma=constants.SIGMA)
    else:
        smoothed_vector_field = vector_field # If smoothing not enabled, use original vector field

    return smoothed_vector_field.reshape(-1,3) # Return smoothed vector field as 1D array, reshape(-1,3) -> automatic row calculation, 3 = RGB columns