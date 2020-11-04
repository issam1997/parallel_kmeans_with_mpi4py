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
    
data = loaded_data["data"]
moyennes = loaded_data["centroids"]
data_size, data_dim = data.shape

t0 = time.time()
count = 0
while True:
    count+=1
    centroids = moyennes.copy()
    clusters = generate_clusters(data, centroids, k)

    for i, cluster in clusters.items():               #items() retourne un tuple (clé, valeur) du dictionnaire
        moyennes[i] = calculate_moyenne(cluster, data_dim)

    error = calculate_error(moyennes, centroids, k)
    
    if error < epsilon:
        break
t1 = time.time()
print(f'le nombre d\'iterations est : {count}')
print(f'time : {t1 - t0}')
# print(__file__.split('.')[0])
if plot:
    plot_clusters('figo', centroids, clusters, data_size)





