import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, shapiro, mannwhitneyu, ttest_ind

# Load JSON data from the file
'''with open('../data/mito_volume_data.json', 'r') as f:
    data = json.load(f)
    extraneous_neuron_ids = ["648518346349538721", "648518346349536769", "648518346349539895", "648518346349538440"]
    for id in extraneous_neuron_ids:
        data.pop(id, None)'''
data = 1 #just temp

def stats(high_volumes, low_volumes):
    # Perform Kolmogorov-Smirnov test
    ks_stat, ks_p_value = ks_2samp(high_volumes, low_volumes)
    print(f"KS Statistic: {ks_stat}, P-value: {ks_p_value}")

    avg_high = np.mean(high_volumes)
    avg_low = np.mean(low_volumes)

    print(f"Average High: {avg_high}, Average Low: {avg_low}")

def OSI_Total_Mito_Volume_BoxPlot_TopBottom10():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import ttest_ind
    import pandas as pd

    # Load and process OSI data
    osi_df = pd.read_csv("../data/osi_data.csv")
    osi_df["osi"] = osi_df["osi"].str.strip("[]").astype(float)
    osi_df["neuron_id"] = osi_df["neuron_id"].astype(str)

    # Define OSI thresholds
    high_osi_ids = set(osi_df[osi_df["osi"] >= 0.6]["neuron_id"])
    low_osi_ids = set(osi_df[osi_df["osi"] < 0.06]["neuron_id"])

    # Extract volumes
    high_volumes = []
    low_volumes = []

    for neuron_id, values in data.items():
        total_mito_volume = values[2]
        
        if neuron_id in high_osi_ids:
            high_volumes.append(total_mito_volume)
        elif neuron_id in low_osi_ids:
            low_volumes.append(total_mito_volume)

    # Calculate means
    mean_high = np.mean(high_volumes) if high_volumes else 0
    mean_low = np.mean(low_volumes) if low_volumes else 0

    # Print summary
    print(f"High OSI (≥ 0.6): {len(high_volumes)} neurons, Mean volume = {mean_high:.2f}")
    print(f"Low OSI (< 0.06): {len(low_volumes)} neurons, Mean volume = {mean_low:.2f}")

    # Plot setup
    plt.figure(figsize=(8, 6))
    plt.boxplot([low_volumes, high_volumes], labels=['Low OSI (< 0.06)', 'High OSI (≥ 0.6)'])

    # Overlay individual points
    positions = [1, 2]
    jitter = 0.1

    plt.scatter(
        np.full(len(low_volumes), positions[0]) + np.random.uniform(-jitter, jitter, len(low_volumes)),
        low_volumes, color='gray', alpha=0.6, label="Low OSI Neurons"
    )
    plt.scatter(
        np.full(len(high_volumes), positions[1]) + np.random.uniform(-jitter, jitter, len(high_volumes)),
        high_volumes, color='red', alpha=0.6, label="High OSI Neurons"
    )

    # Overlay mean values as squares
    plt.scatter(positions[0], mean_low, color='black', marker='s', s=80, label="Mean (Low OSI)")
    plt.scatter(positions[1], mean_high, color='darkred', marker='s', s=80, label="Mean (High OSI)")

    # Labels and formatting
    plt.ylabel("Total Dendritic Mito Volume")
    plt.title("Dendritic Mito Volumes: High vs Low OSI Neurons")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def High_OSI_Dendritic_Mito():
    high_volumes = []
    low_volumes = []

    for values in data.values():
        calcium_activity_level = values[0]
        osi_activity_level = values[1]
        total_mito_volume = values[2]

        if osi_activity_level == "High":
            if calcium_activity_level == 'High':
                high_volumes.append(total_mito_volume)
            else:
                low_volumes.append(total_mito_volume)
    print(len(low_volumes), len(high_volumes))

    plt.figure(figsize=(8, 6))
    plt.boxplot([low_volumes, high_volumes], labels=['Low Calcium', 'High Calcium'])

    # Overlay individual data points using scatter for visibility
    positions = [1, 2]  # x-axis positions for High and Low OSI
    jitter = 0.1  # Small jitter to prevent overlap
   
    plt.scatter(
            np.full(len(low_volumes), positions[0]) + np.random.uniform(-jitter, jitter, len(low_volumes)), 
            low_volumes, c='#E23F44', alpha=1, label="Low Calcium Neurons"
        )

    plt.scatter(
        np.full(len(high_volumes), positions[1]) + np.random.uniform(-jitter, jitter, len(high_volumes)), 
        high_volumes, c='#8b0000', alpha=1, label="High Calcium Neurons"
    )     

    plt.ylabel("Total Dendritic Mito Volume")
    plt.title("Box Plot of High OSI Dendritic Mito Volumes by Calcium Activity Level")
    plt.legend()
    plt.grid()

    plt.show()
    t_stat, p_value = ttest_ind(high_volumes, low_volumes, equal_var=False)  # Welch’s t-test (better for unequal variances)
    print(f"T-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")

def OSI_Normalized_Mito_Near_Spine():
    high_volumes = []
    low_volumes = []

    for values in data.values():
        osi_activity_level = values[1]
        total_mito_volume = values[2]
        spine_volumes = values[3]
            
        normalized_spine_volumes = [v / total_mito_volume for v in spine_volumes]

        if osi_activity_level == "High":
            high_volumes.extend(normalized_spine_volumes)
        else:
            low_volumes.extend(normalized_spine_volumes)

    # Sort data for cumulative probability
    high_volumes.sort()
    low_volumes.sort()

    # Compute cumulative probabilities
    high_cdf = np.linspace(0, 1, len(high_volumes))
    low_cdf = np.linspace(0, 1, len(low_volumes))

    # Plot cumulative probability graph
    plt.figure(figsize=(8, 6))
    plt.plot(high_volumes, high_cdf, label="High OSI", color='r')
    plt.plot(low_volumes, low_cdf, label="Low OSI", color='gray')

    plt.xlabel("Normalized Mitochondrial Volume Near Spines")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Probability of Normalized Mitochondrial Volume Near Spines by OSI Activity Level")
    plt.legend()
    plt.grid(False)

    # Show plot
    plt.show()

    stats(high_volumes, low_volumes)

def High_OSI_Normalized_Mito_Near_Spine():
    high_volumes = []
    low_volumes = []

    for values in data.values():
        calcium_activity_level = values[0]
        osi_activity_level = values[1]
        total_mito_volume = values[2]
        spine_volumes = values[3]
            
        normalized_spine_volumes = [v / 1 for v in spine_volumes]

        if osi_activity_level == "High":
            if calcium_activity_level == 'High':
                high_volumes.extend(normalized_spine_volumes)
            else:
                low_volumes.extend(normalized_spine_volumes)

    # Sort data for cumulative probability
    high_volumes.sort()
    low_volumes.sort()

    # Compute cumulative probabilities
    high_cdf = np.linspace(0, 1, len(high_volumes))
    low_cdf = np.linspace(0, 1, len(low_volumes))

    # Plot cumulative probability graph
    plt.figure(figsize=(8, 6))
    plt.plot(high_volumes, high_cdf, label="High Calcium", color='r')
    plt.plot(low_volumes, low_cdf, label="Low Calcium", color='gray')

    plt.xlabel("Unnormalized Mitochondrial Volume Near Spines")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Probability of High OSI Unnormalized Mitochondrial Volume Near Spines by Calcium Activity Level")
    plt.legend()
    plt.grid(False)

    # Show plot
    plt.show()

    stats(high_volumes, low_volumes)


def OSI_Unnormalized_Mito_Near_Spine_10():
    # Load OSI data
    osi_df = pd.read_csv("../data/osi_data.csv")

    # Clean and convert OSI values
    osi_df["osi"] = osi_df["osi"].str.strip("[]").astype(float)

    sorted_df = osi_df.sort_values("osi", ascending=False)
    top10_ids = list(sorted_df["neuron_id"].head(10))
    bottom10_ids = list(sorted_df["neuron_id"].tail(10))

    high_volumes = []
    low_volumes = []
    print(top10_ids)
    for neuron_id, values in data.items():
        spine_volumes = values[3]
        if int(neuron_id) in top10_ids:
            high_volumes.extend(spine_volumes)
        elif int(neuron_id) in bottom10_ids:
            low_volumes.extend(spine_volumes)

    # Sort data for cumulative probability
    high_volumes.sort()
    low_volumes.sort()

    # Compute cumulative probabilities
    high_cdf = np.linspace(0, 1, len(high_volumes))
    low_cdf = np.linspace(0, 1, len(low_volumes))

    # Plot cumulative probability graph
    plt.figure(figsize=(8, 6))
    plt.plot(high_volumes, high_cdf, label="High OSI", color='r')
    plt.plot(low_volumes, low_cdf, label="Low OSI", color='gray')

    plt.xlabel("Mitochondrial Volume Near Spines")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Probability of Unnormalized Mitochondrial Volume Near Spines by OSI Activity Level")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

    stats(high_volumes, low_volumes)

def New_OSI_Unnormalized_Mito_Near_Spine_10():
    # Load OSI data
    with open("../data/mito_volume_data_r=1.5.json", "r") as f:
        mito_data = json.load(f)

    

    high_volumes = []
    low_volumes = []
    count = 0
    for neuron_id, values in mito_data.items():
        osi = values[0]
        spine_volumes = values[3]
        spine_volumes = [float(v) for v in spine_volumes if v != "NaN"]
        if osi > 0.6:
            high_volumes.extend(spine_volumes)
            count += 1
        elif osi < 0.06:
            low_volumes.extend(spine_volumes)
            count += 1
    print(count)

    # Sort data for cumulative probability
    high_volumes.sort()
    low_volumes.sort()

    # Compute cumulative probabilities
    high_cdf = np.linspace(0, 1, len(high_volumes))
    low_cdf = np.linspace(0, 1, len(low_volumes))

    # Plot cumulative probability graph
    plt.figure(figsize=(8, 6))
    plt.plot(high_volumes, high_cdf, label="High OSI", color='r')
    plt.plot(low_volumes, low_cdf, label="Low OSI", color='gray')

    plt.xlabel("Mitochondrial Volume Near Spines")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Probability of Unnormalized Mitochondrial Volume Near Spines by OSI Activity Level (r=1.5)")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

    stats(high_volumes, low_volumes)
New_OSI_Unnormalized_Mito_Near_Spine_10()