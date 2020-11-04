"""
data_size : nombre d’observations
data_dim : dimension d’une observation
k : nombre de clusters
data : ensemble de données 
myB : un tableau de deux dimentions (data_size, k)
indices : un tableau de deux dimentions (k, Ni) ou Ni le nombre d'observations dans cluster Ci 
centroids : centroïds
partition : une partie du données
rank : rang du processus
size : nombre de processus 

Début    
    répéter :
        centroid := moyenne
        
        pour  i=0 à data_size - 1 faire : 
            myB[i][rank] := euclidean_distance(centroid, data[i])
        
        pour  i=0 à data_size - 1 faire :
            indices[argmin(myB[i])] = i
        
        pour  j=0 à data_dim - 1 faire:
            somme = 0
            nbrElement = 0
            pour  chaque i dans indices[rank]:
                somme += data[i][j]
                nbrElement += 1
            si nbrElement > 0 alors:
                moyenne[j] = somme / nbrElement
            sinon :
                moyenne[j] = 0

        myfin = False
        if euclidean_distance(moyenne, centroid) < epsilon:
            myfin = True
        fin = allreduce(myfin, op=LAND)
        
    jusqua fin == True

fin
"""
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import euclidean_distance


loaded_data = np.load(f"generated_data/2clusters_100000points_2dim.npz")
k = 2
epsilon = 0.00001
max_iter = 10
plot = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t0 = time.time()

data = loaded_data["data"]
centroids = loaded_data["centroids"]
data_size, data_dim = data.shape
# chaque processus se concentre sur un seul centroid
moyenne = centroids[rank]
myB = np.array([[0.]*k for i in range(data_size)])
count = 0

while True:    
    count+=1
    centroid = moyenne.copy()

    # chaque processus mets la distance entre chaque point dans un indice egale a son rank
    for i in range(data_size):
        myB[i][rank] = euclidean_distance(centroid, data[i])
    
    # on fait le reduce pour avoir la distance de chaque point avec chaque centroid    
    Bm = comm.allreduce(myB, op=MPI.SUM)

    # dans B on mets l'indice du distance minimale dans le tableau d'indice i qui est le rank
    indices = [[] for i in range(k)]
    for i in range(data_size):
        indices[np.argmin(Bm[i])].append(i)

    # ici on calcule les nouveau centroid 
    for j in range(data_dim):   
        somme = 0
        nbrElement = 0
        for i in indices[rank]:
            somme += data[i][j]
            nbrElement += 1
        if nbrElement > 0:
            moyenne[j] = somme/nbrElement
        else :
            moyenne[j] = 0

    myfin = euclidean_distance(moyenne, centroid) < epsilon
    fin = comm.allreduce(myfin, op=MPI.LAND)
    if fin or count >= max_iter:
        break 

print(f"le processus {rank} a fait {count} iterations")

t1 = time.time()
if rank == 0:
    print(f"time : {t1 - t0}")

MPI.Finalize()



