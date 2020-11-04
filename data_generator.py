import numpy as np
import math
import random

k = 10
num_points_per_cluster = 10000

def euclidean_distance(point_a, point_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))

def generate_data(k, num_points_per_cluster):
    
    # genearet cluster centers with minimum distance = 4
    centers = np.array([2 * np.random.randn(2) + np.random.rand()*10])
    i = 1
    while i < k:
        center = 2 * np.random.randn(2) + np.random.rand()*10
        d = float('inf')
        for j in range(len(centers)):
            d = min(d, euclidean_distance(center, centers[j]))
        if d >= 3:
            centers = np.vstack((centers, center))
            i += 1
            
    data = np.random.randn(num_points_per_cluster, 2) + centers[0]
    
    # genearate clusters with 200 points each
    for i in range(1, k):
        cluster_data = np.random.randn(num_points_per_cluster, 2) + centers[i]
        data = np.concatenate((data, cluster_data), axis = 0)
    np.random.shuffle(data)
    
    centroids = random.sample(list(data), k) 
            
    np.savez(f'generated_data/{k}clusters_{k*num_points_per_cluster}points_2dim',data=data, centroids=centroids)


generate_data(k, num_points_per_cluster)