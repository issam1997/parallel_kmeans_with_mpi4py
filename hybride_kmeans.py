"""

moyennes et data sont generer dans un ficher 

data_size : nombre d’observations
data_dim : dimension d’une observation
K : nombre de clusters
data : ensemble de données 
error : marge d’erreur
clusters : un dictionnaire de données
centroids : centroïds
partition : une partie du données
rank : rang du processus
size : nombre de processus 
cluster : un tableau pour recuperer le cluster de chaque processus
Début
    partition := get_partition(data, size, rank) 
    moyenne := centroids[rank]
    répéter :
        centroids = allgather(moyenne)
        my_clusters := generate_clusters(partition, centroids, k)    // genere un dictionnaire de clusters
        
        allclusters = allgather(my_clusters)
        pour  i=0 à size - 1 faire :
            cluster.extend(allclusters[i][rang])           
 	    moyenne := calculate_moyenne(cluster, data_dim) 
	    centroid := centroids[rank]
        error := euclidean_distance(moyenne, centroid)  
        error := allreduce(error, opération=max) 

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

    # ici on vas avoir un tableau de dictionnaires 
    # my_clusters = np.array(my_clusters)
    # allclusters = comm.allgather(my_clusters)
    
    # # # # le nombre de processus doit etre egale au nombre du clusters
    # for i in range(size):
    #     cluster.extend(allclusters[i][rank])

    # print(f'rank {rank} cluster ::: {my_clusters}')
    for i in range(k):
        # print(np.array(my_clusters[i]).shape)
        cluster[i] = comm.gather(my_clusters[i], root=i)

    # print(np.array(cluster[rank]).shape)
    # my_cluster = [cluster[rank][i] for i in range(k) if cluster[rank][i]!=[]]
    # print(f' process rank {rank} cluster :::   {my_cluster}')
    
    # my_cluster = np.concatenate(my_cluster)
    # for i in range(k):
    #     my_cluster.extend(cluster[rank][i])
    # my_cluster = cluster[rank]

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

# print(f"le processus {rank} a fait {count} iterations")

t1 = time.time()

if rank == 0:
    # print(f"centroids {centroids}")
    print(f"time : {t1 - t0}")
    # print(f"clusetrs : {clusters}")
    if size >=2 and plot:
        plot_clusters(centroids, clusters, data_size, size)

   



