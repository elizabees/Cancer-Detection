import numpy as np
import random

#initialize centroids randomly
def initialize_centroids(X, k):
    num_samples, num_features = X.shape
    centroids = np.zeros((k, num_features))
    random_indices = np.random.choice(num_samples, k, replace=False)
    centroids = X[random_indices]
    return centroids

#computee euclidean distance between data points and centroids
def compute_distances(X, centroids):
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances

#assigns clusters based on the closest centroids
def assign_clusters(distances):
    return np.argmin(distances, axis=1)

#updates centroids based on the mean of the assigned clusters
def update_centroids(X, clusters, k):
    num_features = X.shape[1]
    centroids = np.zeros((k, num_features))
    for i in range(k):
        points_in_cluster = X[clusters == i]
        if len(points_in_cluster) > 0:
            centroids[i] = np.mean(points_in_cluster, axis=0)
    return centroids

#implements k-means algorithm
def kmeans(X, k, num_iters=100):
    #initialize centroids randomly
    centroids = initialize_centroids(X, k)
    
    for _ in range(num_iters):
        #compute distances between data points and centroids
        distances = compute_distances(X, centroids)
        
        #assign clusters based on the closest centroids
        clusters = assign_clusters(distances)
        
        #update centroids by calculating the mean of assigned clusters
        new_centroids = update_centroids(X, clusters, k)
        
        #if centroids do not change, converged
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, clusters


def main():
    #data set
    X = np.array([
    [2, 3], [3, 2], [4, 4], [5, 1], [3, 3],
    [10, 10], [12, 11], [11, 12], [13, 10], [11, 11]
    ])
    #num clusters
    k = 2

    centroids, clusters = kmeans(X, k)

    print("Final centroids:\n", centroids)
    print("Cluster assignments:", clusters)

if __name__ == "__main__":
    main()