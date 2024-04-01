import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data, n_clusters, linkage='ward'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    model.fit(data_scaled)
    return model.labels_

def fit_modified(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    Z = linkage(data_scaled, 'ward')
    diff = np.diff(Z[:, 2])
    elbow = np.argmax(diff)
    cutoff_distance = Z[elbow, 2]
    model = AgglomerativeClustering(distance_threshold=cutoff_distance, n_clusters=None)
    model.fit(data_scaled)
    return model.labels_, cutoff_distance


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    random_state = 42

    # Generate datasets as per 1.A
    datasets_dict = {
        'nc': datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state),
        'nm': datasets.make_moons(n_samples=100, noise=.05, random_state=random_state),
        'bvv': datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state),
        'add': None,  # This will be generated below as it requires a transformation.
        'b': datasets.make_blobs(n_samples=100, random_state=random_state)
    }

    # Apply the anisotropic transformation to 'add' dataset
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    add = datasets.make_blobs(n_samples=100, random_state=random_state)
    datasets_dict['add'] = (np.dot(add[0], transformation), add[1])
    
    # Perform hierarchical clustering on the datasets with different linkage criteria
    linkages = ['ward', 'complete', 'average', 'single']
    for dataset_name, (dataset_data, _) in datasets_dict.items():
        for linkage in linkages:
            labels = fit_hierarchical_cluster(dataset_data, 2, linkage=linkage)

    dct = answers["4A: datasets"] = labels

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = [""]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    # dct is the function described above in 4.C
    dct = answers["4A: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
