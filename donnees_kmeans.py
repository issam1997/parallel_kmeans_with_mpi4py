"""
moyennes et data sont générer depuis un fichier

data_size : nombre d’observations
data_dim : dimension d’une observation
k : nombre de clusters
data : ensemble de données 
error : marge d’erreur
clusters : un dictionnaire de données
centroids : centroïds
partition : une partie du données
rank : rang du processus
size : nombre de processus 
my_lengths, length : un tableau contenant la longeur de chaque cluster 
my_sums, sums : un tableau contenant la somme des observations dans chaque cluster
Début
    partition := get_partition(data, size, rank)     
    répéter :
        centroids := moyennes
        my_clusters := generate_clusters(partition, centroids, k)    

        pour  i=0 à k - 1 faire : my_lengths[i] := longueur(my_clusters[i])
        lengths := allreduce(my_lengths, opération=sum)
     
        my_sums := calculate_sums(my_clusters, data_dim)                                                                          
        sums := allreduce(my_sums, opération=sum)
        
        pour  i=0 à k - 1 faire :
            pour  j=0 à data_dim - 1 faire:
                si lengths[i] > 0 alors:
                    moyennes[i][j] = sums[i][j] / lengths[i]
                sinon :
                    moyennes[i][j] = 0
       
        error := calculate_error(moyennes, centroids, k)    //calcule l'erreur comme le maximum des distances
        
    jusqua error < epsilon

fin
"""






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
# Partitioning Data

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
        if plot:
            aclusters = comm.gather(clusters, root=0)
        break

t1 = time.time()
# print(f"le processus {rank} a fait {count} iterations")
if rank == 0:
    # print(f"centroids : {centroids}")
    print(f"time : {t1 - t0}")
    if size >=2 and plot :
        clusters = result_clusters(aclusters, k, size)
        plot_clusters('figo.png',centroids, clusters, data_size, size)

MPI.Finalize()




