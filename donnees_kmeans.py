from mpi4py import MPI
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *

loaded_data = np.load("generated_data/10clusters_100000points_2dim.npz")
k = 10
epsilon = 0.00001
max_iter = 10
plot = False

# group the clusters in order to plot results
def result_clusters(aclusters, k, size):
    clusters = [[] for i in range(k)]
    for r in range(k):
        for i in range(size):
            clusters[r].extend(aclusters[i][r])
    return clusters

# plot clusters and centroids


#loaded data contains both precalculated centroids and the generated data 
data = loaded_data["data"]
moyennes = loaded_data["centroids"]
data_size, data_dim = data.shape

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t0 = time.time()

# each process gets its own partion to work on separately
partition = get_partition(data, size, rank)
count = 0
while True:    

    count+=1
    centroids = moyennes.copy()
    my_clusters = generate_clusters(partition, centroids, k)

    # reduce clusters sizes to be able to calculate centroids
    my_lengths = np.array([len(my_clusters[i]) for i in range(k)])
    lengths = comm.allreduce(my_lengths, op=MPI.SUM)
    
    # calculate sum of points on local clusters and then reduce the sums
    my_sums = calculate_sums(my_clusters, data_dim)
    sums = comm.allreduce(my_sums, op=MPI.SUM)
    
    moyennes = [[sums[i][j]/lengths[i] if lengths[i] > 0 else 0 for j in range(data_dim)] for i in range(k)]
    
    error = calculate_error(moyennes, centroids, k)
    if error < epsilon:
        break

t1 = time.time()
# print(f"le processus {rank} a fait {count} iterations")
if rank == 0:
    print(f"time : {t1 - t0}")

MPI.Finalize()




