import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import random

# Load your calcium trace data (adjust the file path as necessary)
with open("../data/calcium_trace.pkl", "rb") as f:
    ca_trace = pickle.load(f)

def activity_levels():
    calcium_nonzero_counts = {}
    ids = list(ca_trace.keys())

    for neuron_id in ids:
        calcium_nonzero_counts[neuron_id] = np.sum(ca_trace[neuron_id]["spike"] > 0)

    nonzero_counts = [*calcium_nonzero_counts.values()]
    nonzero_counts = np.array(nonzero_counts).reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(nonzero_counts)
    cluster_centers = kmeans.cluster_centers_
    clusters = kmeans.predict(nonzero_counts)

    labels = ['Low Activity' if c == cluster_centers.min() else 'High Activity' for c in cluster_centers[clusters]]
    low_activity_levels = {ids[i]: labels[i] for i in range(len(ids)) if labels[i] == 'Low Activity'}
    high_activity_levels = {ids[i]: labels[i] for i in range(len(ids)) if labels[i] == 'High Activity'}

    return low_activity_levels, high_activity_levels    
 
'''
jitter_low = [random.uniform(-0.1, 0.1) for _ in range(len(nonzero_counts))]
jitter_high = [random.uniform(-0.1, 0.1) for _ in range(len(nonzero_counts))]
x_values = [1 + jitter_low[i] if labels[i] == 'Low Activity' else 2 + jitter_high[i] for i in range(len(labels))]

plt.scatter(x_values, nonzero_counts.flatten(), c=clusters, cmap='viridis')
plt.xticks([1, 2], ['Low Activity', 'High Activity'])
plt.ylabel('nonzero_count Values')
plt.title('KMeans Clustering: Low vs High Activity')
plt.scatter([1, 2], cluster_centers, c='red', marker='x', label='Cluster Centers')
plt.legend()
plt.show()'''