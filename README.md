# implementation of kmeans algorithm 

using mpi4py kmeans algorithm is implemented to be executed in parallel

## donnees_kmeans.py
we split data so that each process generates the clusters in a partition and we reduce the results.

## calcul_kmeans.py
each process is charged with one single centroid and has all data, the process then calculates the distance of that centroid with each observation, and 
results are aggregated to find centroids with minimum distances.

## hybrid_kmeans.py
we split data and gather a cluster in each process to calculate new centroids.


