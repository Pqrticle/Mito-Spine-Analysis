import numpy as np
import pandas as pd
import json
import os
import skeletor as sk
from scipy.spatial import cKDTree

def get_branch_polylines_by_length(skeleton, min_length=1, max_length=5, min_nodes=5, max_nodes=30, radius_threshold=2):
    """
    Extract branches from the skeleton based on length, node count, and each node's radius, and return a list of polylines
    along with their corresponding radii for each node.

    Parameters
    ----------
    skeleton : meshparty.skeleton.Skeleton
        The skeleton object with vertices and edges.
    min_length : float
        The minimum branch length in nanometers.
    max_length : float
        The maximum branch length in nanometers.
    min_nodes : int
        The minimum number of nodes in the branch.
    max_nodes : int
        The maximum number of nodes in the branch.
    radius_threshold : float
        The maximum radius of the last node in nanometers.

    Returns
    -------
    polylines : list of np.ndarray
        A list of polylines where each polyline is an array of vertices.
    radii : list of np.ndarray
        A list of radii for each polyline, where each radii array corresponds to the radii of all nodes in the polyline.
    """
    polylines = []
    radii = []

    # Loop over each branch (segment) in the skeleton
    for seg in skeleton.get_segments():
        # Check if the number of nodes in the branch is within the specified range
        if min_nodes <= len(seg) <= max_nodes:
            # Calculate the total length of the branch
            branch_vertices = skeleton.vertices[seg]
            branch_edges = np.diff(branch_vertices, axis=0)
            branch_lengths = np.linalg.norm(branch_edges, axis=1)
            total_length = np.sum(branch_lengths)

            # Get the radii for all nodes in the branch
            node_radii = skeleton.swc.loc[seg, 'radius'].values

            # Check if the branch length and last node's radius are within the specified ranges
            if min_length <= total_length <= max_length and node_radii[-1] < radius_threshold:
                # Append the branch vertices as a polyline and its corresponding node radii
                polylines.append(branch_vertices)
                radii.append(node_radii)

    return polylines, radii

def link_base_synapse(polylines, radii, filt_df_synapse, neuron_id, max_distance_nm=1):
    """
    For each polyline:
    - Finds the closest synapse to the first node.
    - Computes a base point at a radius from the last node.
    - Drops entries where synapse is more than max_distance_nm from the first node.
    
    Returns a DataFrame with:
    base_id, base_x, base_y, base_z,
    post_pos_x_vx, post_pos_y_vx, post_pos_z_vx, synapse_id, distance_to_synapse
    """
    
    neuron_synapses = filt_df_synapse[filt_df_synapse["post_root_id"] == neuron_id].copy()
    if neuron_synapses.empty:
        print(f"No synapses found for neuron {neuron_id}")
        return pd.DataFrame()

    syn_coords = neuron_synapses[["post_pos_x_vx", "post_pos_y_vx", "post_pos_z_vx"]].values
    syn_tree = cKDTree(syn_coords)

    results = []

    for polyline, radius in zip(polylines, radii):
        if len(polyline) < 2:
            continue

        first_node = polyline[0]
        last_node = polyline[-1]
        last_radius = radius[-1]

        # Find closest synapse to the first node
        dist, idx = syn_tree.query(first_node)
        if dist > max_distance_nm:
            continue

        syn_row = neuron_synapses.iloc[idx]

        # Compute base point (radius away from last node toward second-last node)
        second_last_node = polyline[-2]
        direction = second_last_node - last_node
        norm = np.linalg.norm(direction)

        if norm == 0:
            continue  # skip if direction is zero

        direction_unit = direction / norm
        base_point = last_node + last_radius * direction_unit

        results.append({
            "synapse_id": str(int(syn_row["id"])),
            "post_root_id": neuron_id,
            "psd_coords": json.dumps([
                round(syn_row["post_pos_x_vx"], 3),
                round(syn_row["post_pos_y_vx"], 3),
                round(syn_row["post_pos_z_vx"], 3)
            ]),
            "base_coords": json.dumps([
                round(base_point[0], 3),
                round(base_point[1], 3),
                round(base_point[2], 3)
            ]),
            "distance": dist
        })
    
    return pd.DataFrame(results)

def snap_spine_bases(neuron_id, neuron_mesh):
    filt_df_synapse = pd.read_csv("../data/pni_synapses_v185.csv")
    filt_df_synapse["post_pos_x_vx"] *= 0.004
    filt_df_synapse["post_pos_y_vx"] *= 0.004
    filt_df_synapse["post_pos_z_vx"] *= 0.04
    
    skel = sk.skeletonize.by_wavefront(neuron_mesh, origins=None, waves=1, step_size=1)
    sk.post.remove_bristles(skel, los_only=False, inplace=True)
    sk.post.clean_up(skel, inplace=True, theta=1)

    polylines, radii = get_branch_polylines_by_length(skel, min_length=1, max_length=5)
    df_bases = link_base_synapse(polylines, radii, filt_df_synapse, neuron_id, max_distance_nm=1.5)
    print(df_bases)

    output_path = '../data/synapse_table.csv'
    write_header = not os.path.exists(output_path)
    df_bases.to_csv(output_path, mode='a', header=write_header, index=False)
    max_row = df_bases.loc[df_bases['distance'].idxmax()]
    print(max_row)