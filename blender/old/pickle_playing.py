import numpy as np
import random
import matplotlib.pyplot as plt
import pickle




#ACTIVITY CLUSTERING DIAGRAM, SHOWS THE SPIKES OF ALL THE CELLS

# Load your calcium trace data (adjust the file path as necessary)
with open("../data/calcium_trace.pkl", "rb") as f:
    ca_trace = pickle.load(f)

lines = {}
random.seed(1)
ids = random.sample(ca_trace.keys(), 10)

# Create a figure with a 2x5 grid for subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # Adjust figsize as needed
axs = axs.flatten()  # Flatten the 2D array of axes for easy indexing

for i, neuron_id in enumerate(ids):
    np.set_printoptions(threshold=np.inf)
    spike_amplitudes = ca_trace[neuron_id]["spike"]
    #print(spike_amplitudes)
    
    # Count values greater than 5
    above_threshold = np.sum(spike_amplitudes > 5)
    lines[neuron_id] = above_threshold        
    
    # Creating a time axis
    time = np.arange(len(spike_amplitudes))  # Assuming each frame corresponds to 1 unit of time

    # Plotting the spike amplitudes over time for each neuron
    axs[i].plot(time, spike_amplitudes, label='Spike Amplitude', color='red')
    axs[i].set_title(f'Spikes of {neuron_id}')
    axs[i].set_xlabel('Time (frames)')
    axs[i].set_ylabel('Spike Amplitude')
    axs[i].set_xlim(0, len(spike_amplitudes))
    axs[i].set_ylim(0, 100)
    axs[i].axhline(0, color='grey', lw=0.5, ls='--')  # Optional: horizontal line at y=0
    axs[i].grid()
    axs[i].legend()
print(lines)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()  # Display all plots in the specified grid layout