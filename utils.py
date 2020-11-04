"""
Les methodes

"""
import math
import matplotlib.pyplot as plt
import numpy as np

def euclidean_distance(point_a, point_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))

def get_partition(data, comm_size, rank):
    div, mod = divmod(len(data), comm_size)
    counts = [div + 1 if p < mod else div for p in range(comm_size)]
    starts = [sum(counts[: p]) for p in range(comm_size)]
    ends = [sum(counts[: p+1]) for p in range(comm_size)]

    partition = data[starts[rank]:ends[rank]]
    return partition


# returns a dictionary of clusters, cluster with indice i is to store points closest to centroid i
def generate_clusters(partition, centroids, k):
    clusters = {i: [] for i in range(k)}
    for point in partition:
        p = np.argmin([euclidean_distance(point, centroid) for centroid in centroids])
        clusters[p].append(point)
    return clusters

# calcul la somme du points dans chaque clsuter
def calculate_sums(clusters, data_dim):
    sums = []
    for cluster in clusters.values():
        c_sum = [0.] * data_dim
        for point in cluster:
            for j in range(data_dim):
                c_sum[j] += point[j]
        sums.append(c_sum)
    return np.array(sums)


# retourne average c'est le nouveau centroid du cluster
def calculate_moyenne(cluster, data_dim):
    average = [0.] * data_dim
    for point in cluster:
        for j in range(data_dim):
            average[j] += point[j]
    for i in range(data_dim):
        if len(cluster):
            average[i] /= len(cluster) 
    return average

def calc_moyenne(cluster, data_dim):
    average = [0.] * data_dim
    length = 0
    for c in cluster:
        length += len(c)
        for point in c:
            for j in range(data_dim):
                average[j] += point[j]
    for i in range(data_dim):
        if length:
            average[i] /= length
    return average



# calculate error as the maximum euclidean distance between new and previous centroids
def calculate_error(moyennes, centroids, k):
    return max([euclidean_distance(moyennes[i], centroids[i]) for i in range(k)])

def plot_clusters(fig_path, centroids, clusters, data_size, k, comm_size):
    colors = ['g', 'b', 'y', 'c', 'm']

    for i in range(k):
        cluster = np.array(clusters[i])
        plt.title(f'visualisation des résultats pour {data_size} points et {comm_size} processus')
        plt.scatter(cluster[:, 0], cluster[:,1], s=5, c=colors[i])
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=100, c='r')
    plt.savefig(fig_path)