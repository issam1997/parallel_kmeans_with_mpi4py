from mpi4py import MPI
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *



loaded_data = np.load(f"generated_data/4cclusters_2000points_2dim.npz")
k = 4
data = loaded_data["data"]
centroids = loaded_data["centroids"]
data_size, data_dim = data.shape
epsilon = 0.00001
max_iter = 10
plot = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Partitioning the Data
partition = get_partition(data, size, rank)
moyenne = centroids[rank]
t0 = time.time()


count = 0
while True:    
# for _ in range(1):
    count+=1
    centroids = comm.allgather(moyenne)
    cluster = [0]*k
    # retourne un dictionnaire qui contient les clusters de chaque centroids 
    my_clusters = generate_clusters(partition, centroids, k)

    for i in range(k):
        cluster[i] = comm.gather(my_clusters[i], root=i)

    # chaque processus calcule le centroid du cluster generer pour centroid i = rank
    moyenne = calc_moyenne(cluster[rank], data_dim)
    centroid = centroids[rank]    
    error = euclidean_distance(moyenne, centroid)
    
    # on prends le maximum des erreurs calculé sur chaque processus
    error = comm.allreduce(error, op=MPI.MAX)
    if error < epsilon:
        if plot:
            clusters = comm.gather(cluster, root=0)   
        break

print(f"le processus {rank} a fait {count} iterations")

t1 = time.time()

if rank == 0:
    print(f"centroids {centroids}")
    print(f"time : {t1 - t0}")
   

MPI.Finalize()

