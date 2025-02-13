import pandas as pd
import activity_clustering
from scipy.stats import mannwhitneyu

# Load spine data
raw_spine_data = pd.read_csv('../data/pni_synapses_v185.csv')

# Get activity levels
low_activity_neurons, high_activity_neurons = activity_clustering.activity_levels()

# Filter the data for low and high activity neurons
low_activity_df = raw_spine_data[raw_spine_data.post_root_id.isin(low_activity_neurons)]
high_activity_df = raw_spine_data[raw_spine_data.post_root_id.isin(high_activity_neurons)]

# Count occurrences of post_root_id (number of spines per id)
low_activity_spine_counts = low_activity_df.post_root_id.value_counts()
high_activity_spine_counts = high_activity_df.post_root_id.value_counts()

# Perform the Mann-Whitney U test
stat, p_value = mannwhitneyu(low_activity_spine_counts, high_activity_spine_counts, alternative='two-sided')

print(f"Mann-Whitney U test p-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("There is a significant difference in the number of spines between low and high activity neurons.")
else:
    print("There is no significant difference in the number of spines between low and high activity neurons.")