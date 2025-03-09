import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def initialize_centroids(X, k, method='random_samples'):
    if method == 'random_points':
        min_vals, max_vals = X.min(axis=0), X.max(axis=0)
        return np.random.uniform(min_vals, max_vals, size=(k, X.shape[1]))
    elif method == 'random_samples':
        return X[np.random.choice(X.shape[0], k, replace=False)]
    else:
        raise ValueError("Invalid initialization method. Choose 'random_points' or 'random_samples'.")

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def assign_clusters(X, centroids):
    distances = np.array([euclidean_distance(X, centroid) for centroid in centroids])
    labels = np.argmin(distances, axis=0)
    return labels, distances

def balanced_assignment(X, labels, distances, centroids):
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    for i in range(X.shape[0]):
        min_dist = np.min(distances[:, i])
        closest_clusters = np.where(distances[:, i] == min_dist)[0]
        if len(closest_clusters) > 1:
            min_cluster = min(closest_clusters, key=lambda c: cluster_sizes.get(c, 0))
            labels[i] = min_cluster

    return labels

def update_centroids(X, labels, k):
    new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else X[np.random.choice(X.shape[0])] for i in range(k)])
    return new_centroids

def k_means(X, k, max_iter=100, init_method='random_points'):
    centroids = initialize_centroids(X, k, method=init_method)
    for _ in range(max_iter):
        labels, distances = assign_clusters(X, centroids)
        labels = balanced_assignment(X, labels, distances, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Explicación breve del código
# 1. Se generan datos sintéticos con make_blobs.
# 2. Se inicializan los centroides con el método seleccionado ('random_points' o 'random_samples').
# 3. Se asignan los puntos a los centroides más cercanos usando distancia euclidiana.
# 4. Se realiza una asignación balanceada si un punto está equidistante entre dos centroides.
# 5. Se recalculan los centroides hasta que convergen o se alcanzan max_iter iteraciones.
# 6. Se grafican los resultados con los puntos y los centroides marcados en rojo.

# Usuario de github: tostadito33

# Generar datos sintéticos
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Ejecutar el algoritmo con la opción de inicialización deseada
init_method = 'random_points'  # Cambiar a 'random_samples' para otro método
labels, centroids = k_means(X, k=4, max_iter=100, init_method=init_method)

# Graficar resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.title("Clustering con K-Means desde cero")
plt.show()
