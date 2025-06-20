import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import random
import orientation_tuning
import pandas as pd

#change name of the file to calcium_activity.py

with open("../data/calcium_trace.pkl", "rb") as f:
    ca_trace = pickle.load(f)

def kmeans_clustering(ids_list):
    calcium_nonzero_counts = {}
    for neuron_id in ids_list:        
        stimlab, _ = orientation_tuning.get_stimlab_scan_id(neuron_id)
        orientation_mask = ~np.isnan(stimlab)
        calcium_nonzero_counts[neuron_id] = np.sum(ca_trace[neuron_id]["spike"][orientation_mask] > 0)

    nonzero_counts = [*calcium_nonzero_counts.values()]
    nonzero_counts = np.array(nonzero_counts).reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(nonzero_counts)
    cluster_centers = kmeans.cluster_centers_
    clusters = kmeans.predict(nonzero_counts)

    labels = ['Low' if c == cluster_centers.min() else 'High' for c in cluster_centers[clusters]]
    activity_levels = {ids_list[i]: labels[i] for i in range(len(ids_list))}

    return activity_levels 

def classify_activity_levels(): #the clustering function
    osi_data = pd.read_csv('../data/osi_data.csv')
    
    calcium_ids = [int(key) for key in ca_trace]
    high_osi_ids = osi_data[osi_data['osi_activity'] == 'High']['neuron_id'].tolist()
    low_osi_ids = osi_data[osi_data['osi_activity'] == 'Low']['neuron_id'].tolist()
    
    common_high_ids = list(set(calcium_ids).intersection(high_osi_ids))
    common_low_ids = list(set(calcium_ids).intersection(low_osi_ids))

    high_osi_activity_levels = kmeans_clustering(common_high_ids)
    low_osi_activity_levels = kmeans_clustering(common_low_ids)

    print(len(common_high_ids), len(common_low_ids))

    return high_osi_activity_levels, low_osi_activity_levels
    
def sum_spike_probability():
    filtered_synapse_data = pd.read_csv('../data/filtered_synapse_data.csv')

    # Classify activity levels for all neurons
    high_osi_activity_levels, low_osi_activity_levels = classify_activity_levels()

    # Map activity levels to pre and post root IDs
    filtered_synapse_data['pre_high_osi_calcium_activity'] = filtered_synapse_data['pre_root_id'].map(high_osi_activity_levels).fillna('No Calcium Data')
    filtered_synapse_data['pre_low_osi_calcium_activity'] = filtered_synapse_data['pre_root_id'].map(low_osi_activity_levels).fillna('No Calcium Data')
    filtered_synapse_data['post_high_osi_calcium_activity'] = filtered_synapse_data['post_root_id'].map(high_osi_activity_levels).fillna('No Calcium Data')
    filtered_synapse_data['post_low_osi_calcium_activity'] = filtered_synapse_data['post_root_id'].map(low_osi_activity_levels).fillna('No Calcium Data')

    # Save the updated DataFrame to a CSV file
    filtered_synapse_data.to_csv('../data/filtered_synapse_data.csv', index=False, na_rep="NaN")

    activity_data = []
    for neuron_id, calcium_activity in high_osi_activity_levels.items():
        activity_data.append([neuron_id, 'High', calcium_activity])

    for neuron_id, calcium_activity in low_osi_activity_levels.items():
        # Add row for low OSI level
        activity_data.append([neuron_id, 'Low', calcium_activity])

    activity_df = pd.DataFrame(activity_data, columns=['neuron_id', 'osi_activity_level', 'calcium_activity_level'])
    activity_df.to_csv('../data/activity_levels.csv', index=False, na_rep="NaN")


sum_spike_probability()
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
plt.show()
'''