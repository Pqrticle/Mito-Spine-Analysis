import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import random
import orientation_direction_tuning
import pandas as pd

#change name of the file to calcium_activity.py

with open("../data/calcium_trace.pkl", "rb") as f:
    ca_trace = pickle.load(f)

def classify_activity_levels(): #the clustering function
    mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')
    
    calcium_nonzero_counts = {}
    calcium_ids = [int(key) for key in ca_trace]

    ids = mito_data[mito_data.cellid.isin(calcium_ids)].cellid.unique().tolist()

    for neuron_id in ids:        
        stimlab, _ = orientation_direction_tuning.get_stimlab_scan_id(neuron_id)
        orientation_mask = ~np.isnan(stimlab)
        calcium_nonzero_counts[neuron_id] = np.sum(ca_trace[neuron_id]["spike"][orientation_mask] > 0)

    nonzero_counts = [*calcium_nonzero_counts.values()]
    nonzero_counts = np.array(nonzero_counts).reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(nonzero_counts)
    cluster_centers = kmeans.cluster_centers_
    clusters = kmeans.predict(nonzero_counts)

    labels = ['Low Activity' if c == cluster_centers.min() else 'High Activity' for c in cluster_centers[clusters]]
    activity_levels = {ids[i]: labels[i] for i in range(len(ids))}

    jitter_low = [random.uniform(-0.1, 0.1) for _ in range(len(nonzero_counts))]
    jitter_high = [random.uniform(-0.1, 0.1) for _ in range(len(nonzero_counts))]
    x_values = [1 + jitter_low[i] if labels[i] == 'Low Activity' else 2 + jitter_high[i] for i in range(len(labels))]

    plt.scatter(x_values, nonzero_counts.flatten(), c=clusters, cmap='viridis')
    plt.xticks([1, 2], ['Low Activity', 'High Activity'])
    plt.ylabel('nonzero_count Values')
    plt.title('KMeans Clustering: Low vs High Activity')
    plt.scatter([1, 2], cluster_centers, c='red', marker='x', label='Cluster Centers')
    plt.legend()
    plt.show()

    return activity_levels 

def sum_spike_probability(neuron_id):
    filtered_synapse_data = pd.read_csv('../data/filtered_synapse_data.csv')

    #filtered_synapse_data.loc[:, 'pre_root_calcium_probability'] = filtered_synapse_data['pre_root_id'].apply(lambda pre_root_id: np.sum(ca_trace[pre_root_id]["spike"]) if pre_root_id in ca_trace else np.nan)
    #filtered_synapse_data.loc[:, 'post_root_calcium_probability'] = np.sum(ca_trace[neuron_id]["spike"]) if neuron_id in ca_trace else np.nan
    
    matching_rows = filtered_synapse_data['post_root_id'] == neuron_id
    activity_levels = classify_activity_levels()
    filtered_synapse_data.loc[matching_rows, 'pre_activity'] = filtered_synapse_data.loc[matching_rows, 'pre_root_id'].map(activity_levels)
    filtered_synapse_data.loc[matching_rows, 'post_activity'] = filtered_synapse_data.loc[matching_rows, 'post_root_id'].map(activity_levels)

    filtered_synapse_data.to_csv('../data/filtered_synapse_data.csv', index=False)
    matching_rows = filtered_synapse_data[filtered_synapse_data['post_root_id'] == neuron_id]
    print(matching_rows)