from sklearn.cluster import DBSCAN
import numpy as np

# Example list of detected points (x, y)
points = np.array([(100, 120), (110, 130), (115, 125), (200, 250), (210, 260)])

db = DBSCAN(eps=30, min_samples=3).fit(points)
labels = db.labels_

# Extract clusters
clusters = {}
for label, point in zip(labels, points):
    if label != -1:  # Ignore noise points
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(point)

# Find clusters with exactly 3 points
for cluster_id, cluster_points in clusters.items():
    if len(cluster_points) == 3:
        print(f"Cluster {cluster_id} contains exactly 3 points: {cluster_points}")