import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def graph():
    filtered_synapse_data = pd.read_csv('../data/filtered_synapse_data.csv')
    # Filter out rows where post_root_id appears more than once
    filtered_df = filtered_synapse_data[~filtered_synapse_data.duplicated(subset="post_root_id", keep=False)]

    # Split the data based on post_osi_activity values
    high_osi_df = filtered_df[filtered_df["post_osi_activity"] == "High"]
    low_osi_df = filtered_df[filtered_df["post_osi_activity"] == "Low"]

    # Function to compute cumulative probability
    def cumulative_prob_plot(df, ax, title):
        for activity_level in ["High", "Low"]:
            subset = df[df["post_activity"] == activity_level]["total_intersected_volume"]
            sorted_values = np.sort(subset)
            cumulative_prob = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            
            ax.plot(sorted_values, cumulative_prob, label=f"Post Activity: {activity_level}")

        ax.set_xlabel("Total Intersected Volume")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(title)
        ax.legend()

    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    cumulative_prob_plot(high_osi_df, axes[0], "Cumulative Probability (post_osi_activity = High)")
    cumulative_prob_plot(low_osi_df, axes[1], "Cumulative Probability (post_osi_activity = Low)")

    plt.tight_layout()
    plt.show()
