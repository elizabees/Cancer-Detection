import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import os #for data loading

#initialize centroids randomly
def initialize_centroids(X, k):
    num_samples, num_features = X.shape
    centroids = np.zeros((k, num_features))
    random_indices = np.random.choice(num_samples, k, replace=False)
    centroids = X[random_indices]
    return centroids

#compute euclidean distance between data points and centroids
def compute_distances(X, centroids):
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances

#assign clusters based on the closest centroids
def assign_clusters(distances):
    return np.argmin(distances, axis=1)

#update centroids based on the mean of the assigned clusters
def update_centroids(X, clusters, k):
    num_features = X.shape[1]
    centroids = np.zeros((k, num_features))
    for i in range(k):
        points_in_cluster = X[clusters == i]
        if len(points_in_cluster) > 0:
            centroids[i] = np.mean(points_in_cluster, axis=0)
    return centroids

#k-Means algorithm implementation
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

#map clusters to the actual labels
def map_clusters_to_labels(clusters, y_true):
    cluster_to_label = {}
    for cluster in np.unique(clusters):
        true_labels = y_true[clusters == cluster]
        most_common_label = np.bincount(true_labels).argmax()
        cluster_to_label[cluster] = most_common_label
    return np.array([cluster_to_label[cluster] for cluster in clusters])

def main():
    #load data
    data_path = os.path.join('data', 'wdbc.data')
    column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    data = pd.read_csv(data_path, header=None, names=column_names)

    #convert diagnosis column to binary (M = malignant, B = benign)
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

    #drop ID column
    X = data.drop(['ID', 'Diagnosis'], axis=1).values
    y = data['Diagnosis'].values

    #feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #num clusters, M and B
    k = 2

    centroids, clusters = kmeans(X_scaled, k)

    #map clusters back to labels
    y_pred = map_clusters_to_labels(clusters, y)

    #evaluate performance
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Print results
    print("Final centroids:\n", centroids)
    print("Cluster assignments:\n", clusters)

    print(f"Clustering Accuracy: {accuracy:.4f}")
    print(f"Clustering Precision: {precision:.4f}")
    print(f"Clustering Recall: {recall:.4f}")
    print(f"Clustering F1-score: {f1:.4f}")

if __name__ == "__main__":
    main()