
# Principal Component Analysis (PCA):
# * PCA is an unsupervised learning technique for reducing the dimensionality of a large dataset consisting of a large number of features.
# * During dimensionality reduction, PCA retains as much as possible of the variation present in the original dataset.
# * While computing PCA of a dataset of N features, the PCA algorithm can generate maximum N number of principal components (PC) (PC_1...PC_N).
# * Along with data dimensionality reduction, PCA is used across a variety of other applications, for example, exploratory data analysis, data compression, de-noising signal data and many more.
# * We can visualize data upto maximum of 3 dimensions or a dataset containing 3 features. If a dataset has, for example, 20 features, it is not possible to visualize the 20 features in 20 dimensional space.
# * PCA helps us to find the most significant feature in a higher dimensional dataset and makes the data visualization easy in 2D and 3D space.

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt


def data_plotting_2d(x, y):
    plt.figure('X1 vs X2')
    plt.plot(x, y, 'x')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def compute_z_mean(X):
    # find feature mean value
    mean = np.mean(X)
    # find deviation of feature X
    deviation = X - mean
    # standard deviation of the feature
    std_dev = np.std(X)
    # finally calc z mean
    z_mean = deviation / std_dev
    return z_mean


def compute_eigen_values_and_vectors(covariance):
    # compute eigen vectors and values
    eigen_valuess, eigen_vectors = la.eig(covariance)
    # sort eigen values in decreasing order
    arg_indices = eigen_valuess.argsort()
    eigen_valuess = eigen_valuess[arg_indices[::-1]]
    eigen_vectors = eigen_vectors[arg_indices[::-1]]
    
    return eigen_valuess, eigen_vectors


x1 = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
x2 = [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]

data_plotting_2d(x1, x2)


# find standardized x1 and x2
Z1 = compute_z_mean(x1)
print(f'Z1:\n {Z1}')

Z2 = compute_z_mean(x2)
print(f'Z2:\n {Z2}')

Z3 = np.array([Z1.T, Z2.T])
print(f'Z3:\n {Z3}')


# Find Covariance matrix:
covar = np.cov(np.array([x1, x2]))


# Compute Eigenvalue and Eigenvector of the covariance matrix
eigen_valuess, eigen_vectors = compute_eigen_values_and_vectors(covar)
print(f'Eigen values:\n {eigen_valuess}')
print(f'Eigen vectors:\n {eigen_vectors}')


# Compute the percentage variance of the Eigen vectors
# Pencentage variance of an Eigen vector = (Eigen value / sum of Eigen values) * 100
# Pencentage variance of the first Eigen vector
variance_percentage_1 = (eigen_valuess[0] / np.sum(eigen_valuess)) * 100
variance_percentage_1

# Pencentage variance of the first Eigen vector
variance_percentage_2 = (eigen_valuess[1] / np.sum(eigen_valuess)) * 100
variance_percentage_2


# The first PC has extracted most of the information (96.3%). Let's assume that the rest 3.6% of information can be ignored by taking the first PC into consideration.
# Use PC1 set to construct new feature set
feature_vector = eigen_vectors[0:1].T
feature_vector


# New feature calculation
# New feature set = Standardized data * feature vector
# compute new feature set using PC1
new_feature = np.matmul(Z3.T, feature_vector)
print(new_feature)