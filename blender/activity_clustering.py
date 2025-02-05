import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import random
import pandas as pd

#change name of the file to calcium_activity.py


mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')
with open("../data/calcium_trace.pkl", "rb") as f:
    ca_trace = pickle.load(f)

def activity_levels(): #the clustering function
    calcium_nonzero_counts = {}
    calcium_ids = [int(key) for key in ca_trace]

    ids = mito_data[mito_data.cellid.isin(calcium_ids)].cellid.unique().tolist()

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
 

def sum_spike_probability(neuron_id):
    filtered_synapse_data = pd.read_csv('../data/filtered_synapse_data.csv')
    matching_rows = filtered_synapse_data[filtered_synapse_data['post_root_id'] == neuron_id]

    matching_rows.loc[:, 'pre_root_calcium_probability'] = matching_rows['pre_root_id'].apply(lambda pre_root_id: np.sum(ca_trace[pre_root_id]["spike"]) if pre_root_id in ca_trace else 'N/A')
    matching_rows.loc[:, 'post_root_calcium_probability'] = np.sum(ca_trace[neuron_id]["spike"]) if neuron_id in ca_trace else 'N/A'

    filtered_synapse_data.to_csv('../data/filtered_synapse_data.csv', index=False)
    print(matching_rows)
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