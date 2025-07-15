import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def prep_json(radius=1):
    with open(f'../data/mito_volume_data_r={radius}.json', 'r') as f:
        raw_data = json.load(f)

    clean_data = {}
    for neuron_id, values in raw_data.items():
        dendritic_volume = values[2]
        if isinstance(dendritic_volume, str) and dendritic_volume == "NaN":
            continue
        clean_data[neuron_id] = values
    print(f"Number of neurons: {len(clean_data)}")

    return clean_data 

def export_cleaned_mito_data_to_csv(mito_data, output_path="../data/cleaned_mito_data.csv"):
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            "neuron_id",
            "osi",
            "dsi",
            "dendritic_volume",
            "mito_volume",
            "mito_density",
            "synapse_mito_fraction"
        ])

        for neuron_id, vals in mito_data.items():
            osi = vals[0]
            dsi = vals[1]
            dend_vol = vals[2]
            mito_vol = vals[3]
            synapse_mito_vols = vals[4]

            # Validate numeric entries
            if any(isinstance(x, str) and x == "NaN" for x in [osi, dsi, dend_vol, mito_vol]):
                continue

            # Compute mito_density
            mito_density = mito_vol / dend_vol if dend_vol != 0 else np.nan

            # Handle synapse mito data
            cleaned_vols = [float(v) if v != "NaN" else np.nan for v in synapse_mito_vols]
            nonzero_count = np.sum((np.array(cleaned_vols) > 0) & ~np.isnan(cleaned_vols))
            total_valid = np.sum(~np.isnan(cleaned_vols))

            if total_valid == 0:
                synapse_fraction = np.nan
            else:
                synapse_fraction = nonzero_count / total_valid

            # Write row
            writer.writerow([
                int(neuron_id),
                osi,
                dsi,
                dend_vol,
                mito_vol,
                mito_density,
                synapse_fraction
            ])

    print(f"CSV export complete: {output_path}")

def osi_dsi_mito_density(mito_data):
    dsi_list = []
    osi_list = []
    color_ratios = []
    for vals in mito_data.values():
        osi = vals[0]
        dsi = vals[1]
        dend_vol = vals[2]
        mito_vol = vals[3]

        if isinstance(dsi, str) or isinstance(osi, str) or isinstance(dend_vol, str) or isinstance(mito_vol, str):
            continue  # skip corrupted entries

        dsi_list.append(dsi)
        osi_list.append(osi)
        color_ratios.append(mito_vol / dend_vol)

    dsi_arr = np.array(dsi_list)
    osi_arr = np.array(osi_list)
    ratios_arr = np.array(color_ratios)

    # Normalize for colormap
    norm = mpl.colors.Normalize(vmin=np.min(ratios_arr), vmax=np.max(ratios_arr))
    cmap = plt.cm.viridis

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(dsi_arr, osi_arr, c=ratios_arr, cmap=cmap, norm=norm, edgecolor='k', s=80)

    # Aesthetic improvements
    ax.set_xlabel("DSI", fontsize=14)
    ax.set_ylabel("OSI", fontsize=14)
    ax.set_title("Dendritic Mitochondrial Load by Selectivity", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mito Volume / Dend Volume", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Layout and save
    plt.tight_layout()
    plt.show()

def osi_dsi_mito_synapse(mito_data):
    dsi_list = []
    osi_list = []
    avg_mito_synapse_list = []

    for vals in mito_data.values():
        osi = vals[0]
        dsi = vals[1]
        synapse_mito_vols = vals[4]

        # Convert "NaN" to np.nan, then average ignoring them
        cleaned_vols = [float(v) if v != "NaN" else np.nan for v in synapse_mito_vols]
        avg_mito = np.nanmean(cleaned_vols)  # Ignores np.nan values

        dsi_list.append(dsi)
        osi_list.append(osi)
        avg_mito_synapse_list.append(avg_mito)

    dsi_arr = np.array(dsi_list)
    osi_arr = np.array(osi_list)
    avg_arr = np.array(avg_mito_synapse_list)

    # Normalize for colormap
    norm = mpl.colors.Normalize(vmin=np.nanmin(avg_arr), vmax=np.nanmax(avg_arr))
    cmap = plt.cm.viridis

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(dsi_arr, osi_arr, c=avg_arr, cmap=cmap, norm=norm, edgecolor='k', s=80)

    # Aesthetic improvements
    ax.set_xlabel("DSI", fontsize=14)
    ax.set_ylabel("OSI", fontsize=14)
    ax.set_title("Avg Mito Volume Near Synapses by Selectivity", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Avg Mito Volume Near Synapse", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Layout and save
    plt.tight_layout()
    plt.show()

def osi_dsi_fraction_synapses_with_mito(mito_data):
    dsi_list = []
    osi_list = []
    fraction_with_mito_list = []

    for vals in mito_data.values():
        osi = vals[0]
        dsi = vals[1]
        synapse_mito_vols = vals[4]

        if not isinstance(synapse_mito_vols, list):
            continue
        if isinstance(osi, str) or isinstance(dsi, str):
            continue

        cleaned_vols = [float(v) if v != "NaN" else np.nan for v in synapse_mito_vols]
        nonzero_count = np.sum((np.array(cleaned_vols) > 0) & ~np.isnan(cleaned_vols))
        total_valid = np.sum(~np.isnan(cleaned_vols))

        if total_valid == 0:
            continue  # avoid division by zero

        fraction_with_mito = nonzero_count / total_valid

        dsi_list.append(dsi)
        osi_list.append(osi)
        fraction_with_mito_list.append(fraction_with_mito)

    dsi_arr = np.array(dsi_list)
    osi_arr = np.array(osi_list)
    fraction_arr = np.array(fraction_with_mito_list)

    # Normalize for colormap
    norm = mpl.colors.Normalize(vmin=np.nanmin(fraction_arr), vmax=np.nanmax(fraction_arr))
    cmap = plt.cm.viridis

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(dsi_arr, osi_arr, c=fraction_arr, cmap=cmap, norm=norm, edgecolor='k', s=80)

    # Aesthetic improvements
    ax.set_xlabel("DSI", fontsize=14)
    ax.set_ylabel("OSI", fontsize=14)
    ax.set_title("Fraction of Synapses with Mito Volume", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Fraction of Synapses with Mito", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Layout and save
    plt.tight_layout()
    plt.show()

mito_data = prep_json()
export_cleaned_mito_data_to_csv(mito_data)