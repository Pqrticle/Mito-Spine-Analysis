import json
import numpy as np
import matplotlib.pyplot as plt


def graph(high_activity_neurons):
    # Load the JSON data
    data = {}
    with open('../data/jsons/test-2.json', 'r') as file:
        data = json.load(file)

    # Placeholder lists for storing volume data based on activity
    low_activity_volumes = []
    high_activity_volumes = []

    # Separate data based on activity level
    for neuron_id, volumes in data.items():
        # Separate data based on activity level
        if int(neuron_id) in high_activity_neurons:  # Assuming high_activity_neurons is defined
            high_activity_volumes.extend(volumes)  # Add only the first 250 elements for high-activity neurons
        else:
            low_activity_volumes.extend(volumes)  

    # Convert lists to numpy arrays for processing
    low_activity_volumes = np.array(low_activity_volumes)
    high_activity_volumes = np.array(high_activity_volumes)

    # Sort volumes for cumulative calculation
    low_activity_volumes.sort()
    high_activity_volumes.sort()

    # Calculate cumulative probabilities
    low_activity_cumprob = np.arange(1, len(low_activity_volumes) + 1) / len(low_activity_volumes)
    high_activity_cumprob = np.arange(1, len(high_activity_volumes) + 1) / len(high_activity_volumes)

    # Plot the cumulative probability distributions
    plt.figure(figsize=(10, 6))
    plt.plot(low_activity_volumes, low_activity_cumprob, label='Low Activity Neurons', color='blue')
    plt.plot(high_activity_volumes, high_activity_cumprob, label='High Activity Neurons', color='red')

    # Customize the plot
    plt.xlabel('Volume of Intersected Mitochondria')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Probability of Intersected Mitochondrial Volume')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    from scipy.stats import ks_2samp

    # Perform the KS test
    ks_statistic, p_value = ks_2samp(low_activity_volumes, high_activity_volumes)

    # Print the results
    print(f"KS Statistic: {ks_statistic}")
    print(f"P-value: {p_value}")

    # Interpret the result
    if p_value < 0.05:
        print("The difference between the distributions is statistically significant.")
    else:
        print("The difference between the distributions is not statistically significant.")



