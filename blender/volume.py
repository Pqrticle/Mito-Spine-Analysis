import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Open the pickle file in 'rb' (read binary) mode
with open('../data/calcium_trace.pkl', 'rb') as file:
    data = pickle.load(file)

raw_mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')
raw_branch_data = pd.read_csv('../data/distancebinfullstats.csv')

mito_volumes = {}
mito_volume_densities = {}
mito_linear_coverage = {}
compartment_map = {'Axonal': 0, 'Basal': 1, 'Apical': 2}

for neuron_id in data.keys():
    if neuron_id in raw_mito_data.cellid.values:
        mito_df = raw_mito_data.loc[raw_mito_data.cellid == neuron_id]
        for index, mito in mito_df.iterrows():
            compartment = mito.compartment
            volume = mito.mito_vx
            if compartment in compartment_map:
                if neuron_id in mito_volumes:
                    mito_volumes[neuron_id][compartment_map[compartment]] += volume
                else:
                    mito_volumes.setdefault(neuron_id, [0, 0, 0])[compartment_map[compartment]] = volume

for neuron_id in data.keys():
    if neuron_id in raw_branch_data.cellid.values:
        branch_df = raw_branch_data.loc[raw_branch_data.cellid == neuron_id]
        for index, branch in branch_df.iterrows():
            compartment_index = int(branch.complbl-1)
            density = branch.mitovoldensity
            linear_coverage = branch.linearmitocoverage
            if compartment_index in compartment_map.values() and np.isfinite(density) and np.isfinite(linear_coverage):
                if neuron_id in mito_volume_densities:
                    mito_volume_densities[neuron_id][compartment_index] += density
                    mito_linear_coverage[neuron_id][compartment_index] += linear_coverage
                else:
                    mito_volume_densities.setdefault(neuron_id, [0, 0, 0])[compartment_index] = density
                    mito_linear_coverage.setdefault(neuron_id, [0, 0, 0])[compartment_index] = linear_coverage

print(compartment_map)
print(mito_volumes)
print(mito_volume_densities)
print(mito_linear_coverage)

# Prepare data for boxplot
axonal_values = []
basal_values = []
apical_values = []

# Iterate through mito_volume_densities to extract values
for neuron_id, densities in mito_volume_densities.items():
    axonal_values.append(densities[0])
    basal_values.append(densities[1])
    apical_values.append(densities[2])

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Create boxplot for each compartment
plt.boxplot([axonal_values, basal_values, apical_values], 
            labels=['Axonal', 'Basal', 'Apical'], 
            patch_artist=True)

# Overlay individual data points with jitter
# Adding jitter to the x-coordinates
jitter_strength = 0.1  # Adjust this value for more or less jitter
axonal_jittered = np.random.normal(1, jitter_strength, len(axonal_values))
basal_jittered = np.random.normal(2, jitter_strength, len(basal_values))
apical_jittered = np.random.normal(3, jitter_strength, len(apical_values))

# Plot the jittered points
plt.plot(axonal_jittered, axonal_values, 'o', color='black', alpha=0.6)  # Axonal
plt.plot(basal_jittered, basal_values, 'o', color='black', alpha=0.6)    # Basal
plt.plot(apical_jittered, apical_values, 'o', color='black', alpha=0.6)  # Apical

# Set y-axis limits
plt.ylim(0, 30)  # Set the y-axis range from 0 to 50

# Customize the appearance
plt.title('Mito Volume Densities by Compartment')
plt.ylabel('Volume Density')
plt.xlabel('Compartment')
plt.grid(axis='y', linestyle='--')

# Show the plot
plt.tight_layout()
plt.show()