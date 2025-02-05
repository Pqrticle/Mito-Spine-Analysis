import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

raw_distancebin_data = pd.read_csv('../data/distancebinfullstats.csv')
mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')
with open("../data/calcium_trace.pkl", "rb") as f:
    ca_trace = pickle.load(f)

calcium_ids = [int(key) for key in ca_trace]
ids = mito_data[mito_data.cellid.isin(calcium_ids)].cellid.unique().tolist()
calcium_nonzero_counts = {neuron_id: np.sum(ca_trace[neuron_id]["spike"] > 0) for neuron_id in ids}

sorted_neurons = sorted(calcium_nonzero_counts.items(), key=lambda x: x[1])
low_activity_neurons = [neuron for neuron, count in sorted_neurons[:5]]
high_activity_neurons = [neuron for neuron, count in sorted_neurons[-5:]]

print(low_activity_neurons, high_activity_neurons)

# Filter the data for low and high activity cells
low_activity_data = raw_distancebin_data[raw_distancebin_data['cellid'].isin(low_activity_neurons)]
high_activity_data = raw_distancebin_data[raw_distancebin_data['cellid'].isin(high_activity_neurons)]
print(low_activity_data)

from scipy.stats import linregress, norm
import numpy as np

# Remove rows where pathlength is 0
low_activity_data = low_activity_data[low_activity_data['pathlength'] != 0]
high_activity_data = high_activity_data[high_activity_data['pathlength'] != 0]

# Create a new 'mitovol_per_pathlength' variable by dividing 'mitovol' by 'pathlength'
low_activity_data = low_activity_data.dropna(subset=['linearsynapsedensity', 'mitovol', 'pathlength'])
low_x = low_activity_data['linearsynapsedensity']
low_y = low_activity_data['mitovol'] / low_activity_data['pathlength']  # mitovol/pathlength

# Calculate correlation coefficient (r) for low activity cells
low_slope, low_intercept, low_r_value, _, _ = linregress(low_x, low_y)

# Filter out NaN values for high activity data
high_activity_data = high_activity_data.dropna(subset=['linearsynapsedensity', 'mitovol', 'pathlength'])
high_x = high_activity_data['linearsynapsedensity']
high_y = high_activity_data['mitovol'] / high_activity_data['pathlength']  # mitovol/pathlength

# Calculate correlation coefficient (r) for high activity cells
high_slope, high_intercept, high_r_value, _, _ = linregress(high_x, high_y)

# Fisher z-transformation to compare r values
def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

low_z = fisher_z(low_r_value)
high_z = fisher_z(high_r_value)

# Standard error for z-values
low_n = len(low_x)
high_n = len(high_x)
se_diff = np.sqrt(1 / (low_n - 3) + 1 / (high_n - 3))

# Test statistic (z-score)
z_diff = (low_z - high_z) / se_diff

# p-value
p_value = 2 * (1 - norm.cdf(abs(z_diff)))

# Results
print(f"Low Activity Cells - r: {low_r_value:.3f}")
print(f"High Activity Cells - r: {high_r_value:.3f}")
print(f"Difference in r-values (z-score): {z_diff:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < 0.05:
    print("The correlation coefficients are statistically different.")
else:
    print("The correlation coefficients are not statistically different.")
