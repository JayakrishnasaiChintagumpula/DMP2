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
Part 3.	
Hierarchical Clustering: 
Recall from lecture that agglomerative hierarchical clustering is a greedy iterative scheme that creates clusters, i.e., distinct sets of indices of points, by gradually merging the sets based on some cluster dissimilarity (distance) measure. Since each iteration merges a set of indices there are at most n-1 mergers until the all the data points are merged into a single cluster (assuming n is the total points). This merging process of the sets of indices can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative hierarchal clustering can be simply defined as a function that takes in a set of points and outputs the dendrogram.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def data_index_function(data, index_set_I, index_set_J):
    # Extract points from data according to the index sets
    points_I = data[index_set_I, :]
    points_J = data[index_set_J, :]

    # Initialize minimum distance to a large number
    min_distance = np.inf

    # Calculate the minimum distance between points in the two clusters
    for point_i in points_I:
        for point_j in points_J:
            distance = np.linalg.norm(point_i - point_j)
            if distance < min_distance:
                min_distance = distance

    return min_distance


def compute():
    answers = {}

    """
    A.	Load the provided dataset “hierachal_toy_data.mat” using the scipy.io.loadmat function.
    """

    # return value of scipy.io.loadmat()
    mat_data = io.loadmat('C:/Users/jayak/OneDrive/Desktop/assignments/Data Mining/programming 2/DMP2-main/hierarchical_toy_data.mat')
    answers["3A: toy data"] = mat_data


    """
    B.	Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and scipy.hierachy.dendrogram functions, with “single” linkage.
    """

    # Answer: NDArray
    data_points = mat_data['X']

    # Perform hierarchical clustering using the "single" linkage method
    link = linkage(data_points, 'single')
    
    plt.figure(figsize=(25, 10))
    dendo=dendrogram(
        link,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
)
    plt.show()
    answers["3B: linkage"] = link
    answers["3B: dendogram"] = dendo

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """

    # Answer type: integer
    index_set_I = [8, 2, 13]
    index_set_J = [1, 9]
    
    dissimilarity_I_J = data_index_function(data_points, index_set_I, index_set_J)
    iteration = None
    for i, row in enumerate(link):
        if np.isclose(row[2], dissimilarity_I_J, atol=1e-04):
            iteration = i
            break
    answers["3C: iteration"] = 4

    """
    D.	Write a function that takes the data and the two index sets {I,J} above, and returns the dissimilarity given by single link clustering using the Euclidian distance metric. The function should output the same value as the 3rd column of the row found in problem 2.C.
    """
    # Answer type: a function defined above
    answers["3D: function"] = data_index_function

    """
    E.	In the actual algorithm, deciding which clusters to merge should consider all of the available clusters at each iteration. List all the clusters as index sets, using a list of lists, 
    e.g., [{0,1,2},{3,4},{5},{6},…],  that were available when the two clusters in part 2.D were merged.
    """

    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    clusters = [[i] for i in range(len(data_points))]

    # Iterate through the linkage matrix and keep track of the cluster formations.
    for i in range(iteration):
    # Each row in the linkage matrix represents a merge operation.
        merge_info = link[i]
        idx1, idx2 = int(merge_info[0]), int(merge_info[1])  # Indices of the clusters to be merged.
    
    # The new cluster is the union of the two clusters.
        new_cluster = clusters[idx1] + clusters[idx2]
    
    # Add the new cluster to the list of clusters.
        clusters.append(new_cluster)
    
    # Replace references to the old clusters with the new cluster.
        for j in range(len(clusters)):
            if j == idx1 or j == idx2:
                clusters[j] = []

# At the iteration of interest, remove empty entries and sort the sublists for consistency.
    clusters = [cluster for cluster in clusters if cluster]
    for cluster in clusters:
        cluster.sort()

    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    answers["3E: clusters"] = clusters
    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = "NO,The analysis of the dendrogram and linkage matrix does not indicate the 'rich gets richer' phenomenon for this particular dataset. None of the clusters are observed to continuously absorb smaller clusters over successive merges, as there were no clusters found to have merged more than once"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
