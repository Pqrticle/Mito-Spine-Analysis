import pandas as pd
import numpy as np
import json

def match_coords(neuron_id):
    filtered_synapse_data = pd.read_csv('../data/filtered_synapse_data.csv')
    with open(f'../data/spine_base_coords/{neuron_id}-SBC.json', 'r') as file:
        spine_bases = np.array(json.load(file))
    
    matching_rows = filtered_synapse_data[filtered_synapse_data['post_root_id'] == neuron_id]
    print('Initial Table:')
    print(matching_rows)
    if matching_rows.empty:
        return
    psd_coords = np.array(matching_rows['psd_coords'].apply(json.loads).tolist(), dtype=np.float64)

    # Match points using a greedy algorithm
    while len(psd_coords) > 0 and len(spine_bases) > 0:
        distances = np.linalg.norm(psd_coords[:, None, :] - spine_bases[None, :, :], axis=2)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        psd_idx, spine_idx = min_idx

        # Assign coordinates to the DataFrame
        synapse_index = matching_rows.index[psd_idx]
        filtered_synapse_data.loc[synapse_index, 'base_coords'] = json.dumps(spine_bases[spine_idx].tolist())

        # Remove matched points
        psd_coords = np.delete(psd_coords, psd_idx, axis=0)
        spine_bases = np.delete(spine_bases, spine_idx, axis=0)
        matching_rows = matching_rows.drop(synapse_index)

    matching_rows = filtered_synapse_data[filtered_synapse_data['post_root_id'] == neuron_id]
    print(matching_rows)
    filtered_synapse_data.to_csv('../data/filtered_synapse_data.csv', index=False)

    return True


#below is how the table from monil was fixed
'''
# Read the filtered synapse data
filtered_synapse_data = pd.read_csv('../data/simply_filtered_synapse_data.csv')

# Combine the x, y, and z coordinates into a list, divide by 1000, and create a JSON string
filtered_synapse_data['cleft_coords'] = filtered_synapse_data.apply(
    lambda row: json.dumps([row['ctr_pt_x_nm'] / 1000, row['ctr_pt_y_nm'] / 1000, row['ctr_pt_z_nm'] / 1000]), axis=1
)

filtered_synapse_data['psd_coords'] = filtered_synapse_data.apply(
    lambda row: json.dumps([round(row['post_pos_x_vx'] * 0.004, 3), round(row['post_pos_y_vx'] * 0.004, 3), round(row['post_pos_z_vx'] * 0.04, 3)]), axis=1
)
# Drop the original x, y, z coordinate columns
filtered_synapse_data = filtered_synapse_data.drop(columns=['ctr_pt_x_nm', 'ctr_pt_y_nm', 'ctr_pt_z_nm'])
filtered_synapse_data = filtered_synapse_data.drop(columns=['post_pos_x_vx', 'post_pos_y_vx', 'post_pos_z_vx'])
filtered_synapse_data = filtered_synapse_data.drop(columns=['pre_pos_x_vx', 'pre_pos_y_vx', 'pre_pos_z_vx'])
filtered_synapse_data = filtered_synapse_data.drop(columns=['ctr_pos_x_vx', 'ctr_pos_y_vx', 'ctr_pos_z_vx'])
filtered_synapse_data = filtered_synapse_data.drop(columns=['Activity'])


# Optionally, save the updated DataFrame back to a CSV file
filtered_synapse_data.to_csv('../data/simply_filtered_synapse_data.csv', index=False)

# Print the resulting DataFrame
print(filtered_synapse_data)
'''