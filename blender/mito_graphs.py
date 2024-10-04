import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot(selected_neuron_ids):
    boxplot_data = []
    for neuron in selected_neuron_ids:
        df = pd.read_csv(f'../data/csvs/{neuron}-mito_ratios.csv')
        neuron_curvatures = []
        for curvature_values in df['Curvatures']:
            curvature_list = eval(curvature_values)
            neuron_curvatures.extend(curvature_list)
        boxplot_data.append(neuron_curvatures)  # Append the flattened list for the current neuron

        plt.figure(figsize=(12, 6))
        plt.hist(neuron_curvatures, bins=30, alpha=0.5, label=f'Neuron {neuron}', edgecolor='black')

    # Set labels and title
    plt.xlabel('Curvature Values')
    plt.ylabel('Frequency')
    plt.title('Histograms of Curvature Values by Neuron ID')
    plt.legend(title='Neuron ID')

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot2(selected_neuron_ids):
    boxplot_data = []
    for neuron in selected_neuron_ids:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(f'../data/csvs/{neuron}-mito_ratios.csv')
        boxplot_data.append(df['Ratio'].tolist())

    # Create boxplots
    plt.figure(figsize=(12, 6))

    # Create boxplot
    sns.boxplot(data=boxplot_data, showfliers=False)

    # Overlay stripplot to show all points
    sns.stripplot(data=boxplot_data, color='black', alpha=0.6, jitter=True)

    # Set labels and title
    plt.xticks(ticks=range(len(selected_neuron_ids)), labels=selected_neuron_ids)  # Set x-tick labels
    plt.xlabel('Neuron ID')
    plt.ylabel('Ratio Values')
    plt.title('Boxplots of Ratio Values by Neuron ID')

    # Show the plot
    plt.tight_layout()
    plt.show()

# mito id above 0.1 ratio [1243619, 1699893, 2559171]
# mito between 0.05 and 0.1 [1011347, 1441798, 1465269, 1569543, 1580251, 2249701, 784575, 895449]