import time
import numpy as np
import multiprocessing
# from sklearn.externals.joblib import Parallel, delayed
from itertools import combinations


def intra_single_cluster(subX, metric):
    """Calculate the mean intra-cluster distance for samples in subX
    Parameters
    ----------
    subX:     a block of samples, array [n_samples_subX, n_features]
    metric: a function calculate pairwise distance, function
    Returns
    -------
    intra_d_subX: array [n_samples_subX], mean distance from sample i to all other samples within the same cluster, for i in subX
    """
    intra_d_subX = np.array(
                        [np.mean(
                            [metric(subX[i],subX[j]) for j in range(subX.shape[0]) if j!=i]) 
                         for i in range(subX.shape[0])])
    return intra_d_subX


def intra_single_job(arguments): # X, label_job, labels, metric, intra_d
    X, label_job, labels, metric, intra_d = arguments
    for label in label_job:
        indices = np.where(labels==label)[0]
        intra_d[indices] = intra_single_cluster(X[indices], metric)
    return intra_d
        
    
def intra_cluster_distances_parallel(X, labels, metric, n_jobs):
    """Calculate the intra-cluster distance for each sample in X by block 
    Parameters
    ----------
    X:      a block of samples, array [n_samples, n_features]
    metric: a function calculate pairwise distance, function
    Returns
    -------
    intra_d: intra cluster distance for each sample in X
    """

    label_jobs = np.array_split(np.unique(labels), n_jobs)
    intra_d = np.zeros(labels.size, dtype=float)
    
    pool = multiprocessing.Pool(n_jobs)
    arguments = [(X, label_job, labels, metric, intra_d) for label_job in label_jobs]
    values = pool.map(intra_single_job, arguments)  
#     values = Parallel(n_jobs=n_jobs)(
#             delayed(intra_single_job)
#                 (X, label_job, labels, metric)
#                 for label_job in label_jobs)
    for values_ in values:
        intra_d = np.maximum(intra_d, values_)
    return intra_d


def inter_single_cluster_pair(subX_a, subX_b, metric):
    """
    Parameters
    ----------
    subX: array[n_subX_a, n_features]
    subY: array[n_subX_b, n_features]
    metric: function to calculate pairwise distance between two samples
    returns 
    -------
    dist_a: array[n_subX_a], distance of array a 
    dist_b: array[n_subX_b]
    """
    dist_matrix = np.zeros([subX_a.shape[0], subX_b.shape[0]])
    for i in range(subX_a.shape[0]):
        for j in range(subX_b.shape[0]):
            dist_matrix[i][j] = metric(subX_a[i], subX_b[j])
            
    dist_a = np.mean(dist_matrix, axis=1)
    dist_b = np.mean(dist_matrix, axis=0)
    return dist_a, dist_b


def inter_single_job(arguments): # X, label_job, labels, metric, inter_d)
    X, label_job, labels, metric, inter_d = arguments
    for label_pair in label_job:
        label_a, label_b = label_pair
        indices_a = np.where(labels == label_a)[0]
        indices_b = np.where(labels == label_b)[0]
        value_a, value_b = inter_single_cluster_pair(X[indices_a], X[indices_b], metric)
        inter_d[indices_a] = np.minimum(value_a, inter_d[indices_a])
        inter_d[indices_b] = np.minimum(value_b, inter_d[indices_b])
    return inter_d


def inter_cluster_distances_parallel(X, labels, metric, n_jobs):
    """Calculate the inter cluster distance for each sample in X
    Parameters
    ----------
    X:      a block of samples, array [n_samples, n_features]
    metric: a function calculates pairwise distance, function
    Returns
    -------
    inter_d: inter cluster distance for each sample in X
    """
    assert(X.shape[0]==labels.size)
    label_combinations = [(label_a, label_b) for label_a, label_b in combinations(np.unique(labels), 2)]
    label_jobs = np.array_split(label_combinations, n_jobs)

    inter_d = np.empty(labels.size, dtype=float)
    inter_d.fill(np.inf)
    
    pool = multiprocessing.Pool(n_jobs)
    arguments = [(X, label_job, labels, metric, inter_d) for label_job in label_jobs]
    values = pool.map(inter_single_job, arguments)
#     values = Parallel(n_jobs=n_jobs)(
#                 delayed(inter_blah)
#                 (X, label_job, labels, metric, inter_d) 
#                 for label_job in label_jobs)

    for value_ in values:
        inter_d = np.minimum(value_, inter_d)
    return inter_d


def silhouttee_score_parallel(X, labels, metric, n_jobs):
    # cehck 2 <= n_labels <= n_samples - 1.
    if len(np.unique(labels)) > len(labels) or len(np.unique(labels))<=1 or X.shape[0]!=labels.shape[0]:
        raise ValueError('input not right')
#     assert(X.shape[0]==labels.shape[0])
    t1 = time.time()
    intra_d = intra_cluster_distances_parallel(X, labels, metric, n_jobs)
    t2 = time.time()
    print(f'intra_distance finish in : {t2 - t1}s')
    inter_d = inter_cluster_distances_parallel(X, labels, metric, n_jobs)
    t3 = time.time()
    print(f'inter_distance finish in: {t3 - t2}s')
    sil_score = np.mean(np.nan_to_num((inter_d - intra_d)/np.maximum(intra_d, inter_d))) # np.maximum take two arrays and compute their element-wise maximum.
    print(f'silhouette_score_parallel finish in: {t3 - t2}s')
    return np.mean(np.nan_to_num(intra_d)), np.mean(inter_d), sil_score
