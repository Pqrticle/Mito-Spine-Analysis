import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp

# Load the JSON data
with open('../data/jsons/mito-spine-volumes.json', 'r') as file:
    mito_spine_volumes = json.load(file)

with open('../data/jsons/all-total-mito-volumes.json', 'r') as file:
    total_volumes = json.load(file)

with open('../data/jsons/all-neuron-volumes.json', 'r') as file:
    neuron_volumes = json.load(file)

def intersected_mito_volume(high_activity_neurons):
    low_activity_volumes = []
    high_activity_volumes = []

    # Separate data based on activity level
    for neuron_id, volumes in mito_spine_volumes.items():
        # Separate data based on activity level
        new_volumes = []
        total_volume = total_volumes[neuron_id]
        for volume in volumes:
            if volume != 0:
                new_volume = volume / total_volume
                new_volumes.append(new_volume)
        if int(neuron_id) in high_activity_neurons:
            high_activity_volumes.extend(new_volumes)
        else:
            low_activity_volumes.extend(new_volumes)  

    # Convert lists to numpy arrays for processing
    low_activity_volumes = np.array(low_activity_volumes)
    high_activity_volumes = np.array(high_activity_volumes)
    low_activity_volumes.sort()
    high_activity_volumes.sort()

     # Perform the KS test
    ks_statistic, p_value = ks_2samp(low_activity_volumes, high_activity_volumes)
    
    print(f"KS Statistic: {ks_statistic}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("The difference between the distributions is statistically significant.")
    else:
        print("The difference between the distributions is not statistically significant.")

    # Calculate cumulative probabilities
    low_activity_cumprob = np.arange(1, len(low_activity_volumes) + 1) / len(low_activity_volumes)
    high_activity_cumprob = np.arange(1, len(high_activity_volumes) + 1) / len(high_activity_volumes)

    # Plot the cumulative probability distributions
    plt.figure(figsize=(10, 6))
    plt.plot(low_activity_volumes, low_activity_cumprob, label='Low Activity Neurons', color='blue')
    plt.plot(high_activity_volumes, high_activity_cumprob, label='High Activity Neurons', color='red')

    # Customize the plot
    plt.xlabel('Intersected Mito Volume per Total Dendritic Volume')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Probability of Intersected Mito Volume per Total Dendritic Volume')
    plt.legend()
    plt.grid(True)
    plt.show()


def total_mito_volume(high_activity_neurons):
    low_activity_volumes = []
    high_activity_volumes = []

    # Separate data based on activity level
    for neuron_id, volume in total_volumes.items():
        if int(neuron_id) in high_activity_neurons:  # Assuming high_activity_neurons is defined
            high_activity_volumes.append(volume )#/ neuron_volumes[neuron_id])
        else:
            low_activity_volumes.append(volume )#/ neuron_volumes[neuron_id])

    # Sort volumes for cumulative probability calculations
    low_activity_volumes = np.sort(low_activity_volumes)
    high_activity_volumes = np.sort(high_activity_volumes)

    # Calculate cumulative probabilities
    low_activity_cumprob = np.arange(1, len(low_activity_volumes) + 1) / len(low_activity_volumes)
    high_activity_cumprob = np.arange(1, len(high_activity_volumes) + 1) / len(high_activity_volumes)

    # Plot the cumulative probability distributions
    plt.figure(figsize=(10, 6))
    plt.plot(low_activity_volumes, low_activity_cumprob, label='Low Activity Neurons', color='blue')
    plt.plot(high_activity_volumes, high_activity_cumprob, label='High Activity Neurons', color='red')

    # Customize the plot
    plt.xlabel('Total Dendritic Volume')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Probability of Total Dendritic Volume per Neuron')
    plt.legend()
    plt.grid(True)
    plt.show()


def chi_square_test(high_activity_neurons):
    low_activity_zero_vals = 0
    low_activity_nonzero_vals = 0
    high_activity_zero_vals = 0
    high_activity_nonzero_vals = 0

    # Separate data based on activity level
    for neuron_id, volumes in mito_spine_volumes.items():
        zero_vals = np.count_nonzero(np.array(volumes) == 0)
        nonzero_vals = np.count_nonzero(volumes)
        normalized_zero_vals = zero_vals  # / len(volumes)
        normalized_nonzero_vals = nonzero_vals # / len(volumes)
        
        if int(neuron_id) in high_activity_neurons:
            high_activity_zero_vals += normalized_zero_vals
            high_activity_nonzero_vals += normalized_nonzero_vals
        else:
            low_activity_zero_vals += normalized_zero_vals
            low_activity_nonzero_vals += normalized_nonzero_vals

    # Create a contingency table for the chi-square test
    contingency_table = [
        [high_activity_zero_vals, high_activity_nonzero_vals],
        [low_activity_zero_vals, low_activity_nonzero_vals]
    ]

    # Perform the chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(contingency_table)
    print(f"Chi-square statistic: {chi2}")
    print(f"p-value: {p}")
    print(f"Degrees of freedom: {dof}")
    print("Expected frequencies:")
    print(expected)